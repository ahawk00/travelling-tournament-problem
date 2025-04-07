# import numpy as np
import random
import pandas as pd
import csv
import os
import timeit
import math
import numpy as np
import copy
from multiprocessing import Pool
from collections import defaultdict, Counter
import argparse
import re



class InitSchedule:
    def __init__(self, n_teams, schedule_type):
        self.n_teams = n_teams
        self.schedule_type = schedule_type
        self.rounds = (n_teams - 1) * 2
        self.row_schedule = self.generate_random_row_first()

    def generate_random_row_first(self):
        schedule_matrix = [[None] * self.n_teams for _ in range(self.rounds)]

        teams = list(range(-self.n_teams, self.n_teams + 1))
        teams.remove(0)

        for round in range(self.rounds):
            teams_to_pick = teams.copy()
            for team in range(self.n_teams):
                if schedule_matrix[round][team] == None:
                    team += 1
                    teams_to_pick.remove(team)  # remove current team
                    teams_to_pick.remove(-team)
                    opponent = random.choice(teams_to_pick)  # randomly choose opponent
                    teams_to_pick.remove(opponent)
                    teams_to_pick.remove(-opponent)
                    if opponent > 0:  # home game of opponent
                        schedule_matrix[round][team - 1] = opponent
                        schedule_matrix[round][opponent - 1] = -team
                    else:
                        schedule_matrix[round][team - 1] = opponent
                        schedule_matrix[round][abs(opponent) - 1] = team

        return schedule_matrix

    def get_schedule(self):
        if self.schedule_type == "row":
            return self.row_schedule
        else:
            return self.column_schedule


class SimulatedAnnealing:
    def __init__(self, init_schedule, distance_matrix, schedule_type, objective_focus, temp=1000, alpha=0.9,
                 end_temp=0.00001, iterations=100000):
        self.init_temp = temp
        self.end_temp = end_temp
        self.temp = temp
        self.alpha = alpha
        self.iterations = iterations

        # random schedule
        self.distance_matrix = distance_matrix
        self.schedule_type = schedule_type
        self.objective_focus = objective_focus
        self.n_teams = len(init_schedule[0])  # n_teams
        self.rounds = len(init_schedule)  # rounds

        # inital
        self.initial_schedule = init_schedule
        self.initial_distance = self.calculate_travel_distance(init_schedule, distance_matrix)
        self.initial_violations = self.row_check_constraints_violated(init_schedule)[0]

        self.best_schedule = init_schedule
        self.best_distance = self.calculate_travel_distance(init_schedule, distance_matrix)
        self.best_violations = self.row_check_constraints_violated(init_schedule)[0]

        self.curr_schedule = init_schedule
        self.curr_distance = self.calculate_travel_distance(init_schedule, distance_matrix)
        self.curr_violations = self.row_check_constraints_violated(init_schedule)[0]

        self.schedule_before_repair = init_schedule

        self.accepted_mutations = 0
        self.nr_schedule_absolute_difference = 0
        self.total_schedule_absolute_difference = 0

    def calculate_travel_distance(self, schedule, distance_matrix):
        """Computes the total travel distance for a given schedule."""
        dist_per_team = [0] * self.n_teams
        total_distance = 0

        for team_idx in range(self.n_teams):
            prev_opponent, prev_location = team_idx, 'home'

            for round in range(self.rounds):
                opponent = schedule[round][team_idx]
                if opponent > 0:
                    if prev_location == "home":
                        distance = distance_matrix[abs(opponent) - 1, team_idx]
                    else:
                        distance = distance_matrix[abs(opponent) - 1, prev_opponent]
                    total_distance += distance
                    dist_per_team[team_idx] += distance
                    prev_opponent, prev_location = abs(opponent) - 1, "away"

                else:
                    if prev_location == "home":
                        distance = 0
                    else:
                        distance = distance_matrix[prev_opponent, team_idx]
                    total_distance += distance
                    dist_per_team[team_idx] += distance
                    prev_opponent, prev_location = abs(opponent) - team_idx, "home"

        return total_distance, [int(d) for d in dist_per_team]

    def row_check_constraints_violated(self, schedule):
        max_streak = 3

        violations = {"total": 0, "double_round_robin": 0, "noRepeat": 0, "maxStreak": 0}
        match_count = {}  # track how many times each matchup has occurred
        last_opponent, last_location = [-1] * self.n_teams, [None] * self.n_teams  # store last opponent per team / "home" or "away"
        home_streaks, away_streaks = [0] * self.n_teams, [0] * self.n_teams

        required_matches = set()
        for team_idx in range(self.n_teams):
            for opponent_idx in range(team_idx + 1, self.n_teams):  # Only generate unique pairs
                required_matches.add(frozenset([team_idx, opponent_idx]))  # Team X vs Team Y (unordered)

        # to generate the home and away matches, we need both (team_idx, opponent_idx) and (opponent_idx, team_idx)
        match_pairs = []
        for match in required_matches:
            team1, team2 = list(match)
            # Adding both (team1 vs team2) and (team2 vs team1)
            match_pairs.append((team1, team2))
            match_pairs.append((team2, team1))

        required_games = set(match_pairs)
        matches_more_than_twice = []
        round_duplicate_matches = defaultdict(list)
        round_team_positions = defaultdict(list)  # team → list of columns it appears in
        directional_match_count = {}

        for round_idx, round_matches in enumerate(schedule):
            teams_played = set()
            home, away = None, None

            for team_idx in range(self.n_teams):
                opponent = round_matches[team_idx]

                home, away = ("opponent", "team") if opponent > 0 else ("team", "opponent")
                abs_opponent = abs(opponent) - 1  # convert to zero-based index

                # drr - check for mutual match consistency (column only)
                expected_response = -(team_idx + 1) if opponent > 0 else team_idx + 1
                actual_response = schedule[round_idx][abs_opponent]
                if actual_response != expected_response:
                    violations["double_round_robin"] += 1
                    violations["total"] += 1

                # team plays once per round
                if abs_opponent in teams_played:
                    # print("played more than once")
                    violations["double_round_robin"] += 1
                    violations["total"] += 1
                    round_team_positions[abs_opponent].append(team_idx)

                teams_played.add(abs_opponent)

                # double round-robin constraint (each matchup occurs twice)
                match = tuple([abs_opponent, team_idx])
                match_count[match] = match_count.get(match, 0) + 1
                # ff a match occurs more than twice, it's a violation
                if match_count[match] > 2:
                    # increment violation count once for each match exceeds two occurrence
                    if match not in matches_more_than_twice:
                        violations["double_round_robin"] += 1
                        violations["total"] += 1
                    matches_more_than_twice.append(match)

                if away == "opponent":
                    ordered_match = tuple([team_idx, abs_opponent])  # ensure ordered pairing
                else:
                    ordered_match = tuple([abs_opponent, team_idx])
                # remove the match from required matches since it's now played
                if ordered_match in required_games:
                    required_games.remove(ordered_match)

                # remove the match from required matches since it's now played
                if ordered_match in required_games:
                    required_games.remove(ordered_match)
                elif match_count[match] > 2 and ordered_match not in required_games and ordered_match not in \
                        round_duplicate_matches[round_idx]:
                    round_duplicate_matches[round_idx].append(ordered_match)

                # no Repeat (no consecutive matches against the same team)
                if last_opponent[team_idx] == abs_opponent:
                    violations["noRepeat"] += 1
                    violations["total"] += 1

                last_opponent[team_idx] = abs_opponent

                # max Streak (No more than `max_streak` consecutive home/away games)
                if opponent > 0:  # home game
                    if last_location[team_idx] == "home":
                        home_streaks[team_idx] += 1
                    else:
                        home_streaks[team_idx] = 1
                    away_streaks[team_idx] = 0
                    last_location[team_idx] = "home"

                else:  # away game
                    if last_location[team_idx] == "away":
                        away_streaks[team_idx] += 1
                    else:
                        away_streaks[team_idx] = 1
                    home_streaks[team_idx] = 0
                    last_location[team_idx] = "away"

                if home_streaks[team_idx] > max_streak or away_streaks[team_idx] > max_streak:
                    violations["maxStreak"] += 1
                    violations["total"] += 1

        # ddr: every team must play every other team exactly twice (one home, one away)
        remaining_matches = len(required_games)
        violations["double_round_robin"] += remaining_matches
        violations["total"] += remaining_matches

        return violations, round_duplicate_matches, round_team_positions, required_games, directional_match_count

    def row_mutation(self, schedule):
        '''Swap 2 teams row wise'''
        # swap team row
        # randomly choose round
        rand_round = random.choice([0, self.rounds - 1])

        new_schedule = copy.deepcopy(schedule)

        # randomly choose 2 teams
        indices = random.sample(range(self.n_teams), 2)
        rand_team1_element = new_schedule[rand_round][indices[0]]
        rand_team2_element = new_schedule[rand_round][indices[1]]

        # check if the team playing itself
        while abs(rand_team1_element) == indices[1] + 1 or abs(rand_team2_element) == indices[0] + 1:
            indices = random.sample(range(self.n_teams), 2)
            rand_team1_element = new_schedule[rand_round][indices[0]]
            rand_team2_element = new_schedule[rand_round][indices[1]]

        # swap the teams
        new_schedule[rand_round][indices[0]] = rand_team2_element
        new_schedule[rand_round][indices[1]] = rand_team1_element

        mutated_distance = self.calculate_travel_distance(new_schedule, self.distance_matrix)
        mutated_violations, *_ = self.row_check_constraints_violated(new_schedule)

        return schedule, mutated_distance, mutated_violations

    def acceptance_prob(self, mutated_obj_measure, curr_obj_measure):
        '''
        Calculates the acceptance probability.
        Based on the difference of the new and current objective measure and the current temp
        '''
        return math.exp(-abs(mutated_obj_measure[0] - curr_obj_measure[0]) / self.temp)

    def accept(self, mutated_schedule, mutated_distance, mutated_violations):
        '''
        Determines acceptance of the new candidate.
        Based on temperature and difference of new candidate and current best candidate
        '''
        if mutated_distance <= self.curr_distance:
            self.curr_schedule, self.curr_distance, self.curr_violations = mutated_schedule, mutated_distance, mutated_violations
            self.accepted_mutations += 1
            if mutated_distance < self.best_distance:
                self.best_schedule, self.best_distance, self.best_violations = self.curr_schedule, self.curr_distance, self.curr_violations

        else:
            if random.random() < self.acceptance_prob(mutated_distance, self.curr_distance):
                self.curr_schedule, self.curr_distance, self.curr_violations = mutated_schedule, mutated_distance, mutated_violations

    def update_temp(self, itr_nr, max_itr, cooling_type):
        if cooling_type == 'linear_reheat':
            division = max_itr / 10
            t_0 = 1000 * 0.5 ** (int(itr_nr / division))
            self.temp = t_0 - t_0 / division * (itr_nr - (int(itr_nr / division) * division))
        if cooling_type == 'geometric':
            self.temp = self.temp * 0.9999
        if cooling_type == 'stairs':  # 500.05
            t_step = 1000 / 10
            self.temp = t_step * (10 - int(itr_nr / (itr_nr / 10)))
        if cooling_type == 'GG_50':
            self.temp = 50 / (np.log(itr_nr + 1) if itr_nr > 0 else 0.01)
        if cooling_type == 'linear':  # 500
            self.temp = self.init_temp * (1 - (self.step / self.max_step))

    def schedule_absolute_difference(self, schedule1, schedule2):
        arr1 = np.array(schedule1)
        arr2 = np.array(schedule2)
        difference = np.sum(arr1 != arr2)
        self.nr_schedule_absolute_difference = difference
        return difference

    def directional_matches(self, schedule):
        directional_matches = defaultdict(list)  # (home, away) → list of rounds it occurred

        for round_idx, round_data in enumerate(schedule):
            for team_idx, opponent_val in enumerate(round_data):
                opponent_idx = abs(opponent_val) - 1
                if team_idx < opponent_idx:  # avoid counting both (A vs B) and (B vs A)
                    if opponent_val > 0:
                        match = (team_idx, opponent_idx)  # team is home
                    else:
                        match = (opponent_idx, team_idx)  # opponent is home
                    directional_matches[match].append(round_idx)

        # find directional matches that occurred more than once
        repeated_same_direction = {match: rounds for match, rounds in directional_matches.items() if len(rounds) > 1}

        return repeated_same_direction

    def smart_round_swaps(self, schedule, violation_rounds, violation_type="streak"):
        round_counts = Counter(violation_rounds)
        ordered_rounds = [r for r, _ in round_counts.most_common()]
        random.shuffle(ordered_rounds)

        for i in range(0, len(ordered_rounds) - 1, 2):
            r1, r2 = ordered_rounds[i], ordered_rounds[i + 1]
            temp_schedule = [row[:] for row in schedule]
            temp_schedule[r1], temp_schedule[r2] = temp_schedule[r2], temp_schedule[r1]

            violations_after, *_ = self.row_check_constraints_violated(temp_schedule)

            if violation_type == "streak" and violations_after["maxStreak"] < self.curr_violations["maxStreak"]:
                schedule[r1], schedule[r2] = schedule[r2], schedule[r1]

            elif violation_type == "repeat" and violations_after["noRepeat"] < self.curr_violations["noRepeat"]:
                schedule[r1], schedule[r2] = schedule[r2], schedule[r1]

        return schedule

    def row_constraint_repair(self, schedule):
        _, round_duplicate_matches, _, _, _ = self.row_check_constraints_violated(schedule)

        for round_idx, duplicate_matches in round_duplicate_matches.items():
            # remove duplicate matches in this round
            unique_matches = []
            seen = set()
            for m in duplicate_matches:
                if m not in seen:
                    seen.add(m)
                    unique_matches.append(m)

            if len(unique_matches) < 2:
                continue  # need at least 2 unique matches to attempt a swap

            match1 = unique_matches[0]
            match2 = unique_matches[1]

            # print(f"match1: {match1}, match2: {match2}")

            match1_val1 = schedule[round_idx][match1[0]]
            match1_val2 = schedule[round_idx][match1[1]]

            match2_val1 = schedule[round_idx][match2[0]]
            match2_val2 = schedule[round_idx][match2[1]]

            # swap opponents of duplicate in same row in schedule
            schedule[round_idx][match1[0]], schedule[round_idx][match2[1]] = match2_val1, match1_val2
            schedule[round_idx][match2[0]], schedule[round_idx][match1[1]] = match1_val1, match2_val2

        # ensures correct home/away assignment
        repeated_directional = self.directional_matches(schedule)

        for (home_team, away_team), rounds in repeated_directional.items():
            # flip the second occurrence to alternate home/away
            if len(rounds) >= 2:
                round_to_flip = rounds[1]

                # flip sign of home_team and away_team's values to reverse direction
                if schedule[round_to_flip][home_team] == away_team + 1:
                    schedule[round_to_flip][home_team] = -(away_team + 1)
                    schedule[round_to_flip][away_team] = home_team + 1
                elif schedule[round_to_flip][home_team] == -(away_team + 1):
                    schedule[round_to_flip][home_team] = away_team + 1
                    schedule[round_to_flip][away_team] = -(home_team + 1)

        # violations after directional fixes
        violations, _, _, _, _ = self.row_check_constraints_violated(schedule)

        # round swaps for streak violations
        _, _, _, _, streak_violations = self.row_check_constraints_violated(schedule)
        schedule = self.smart_round_swaps(schedule, list(streak_violations), violation_type="streak")

        # re-evaluate after fixing streaks and now fix repeat violations
        _, _, _, _, repeat_violations = self.row_check_constraints_violated(schedule)
        schedule = self.smart_round_swaps(schedule, list(repeat_violations), violation_type="repeat")

        violations, *_ = self.row_check_constraints_violated(schedule)
        distance = self.calculate_travel_distance(schedule, self.distance_matrix)
        self.curr_schedule, self.curr_violations, self.curr_distance = schedule, violations, distance

        if distance < self.best_distance:
            self.best_schedule, self.best_violations, self.best_distance = schedule, violations, distance

        return schedule, violations

    def anneal(self, result_file, cooling_type):
        # # rand schedule + distance
        schedule = self.best_schedule

        result_writer = csv.writer(result_file)
        result_writer.writerow(
            ['iteration', 'n', 'schedule_type', 'distance', 'distance_per_team', 'violations', 'schedule', 'schedule_absolute_diff'])
        result_writer.writerow(
            [1, self.n_teams, self.schedule_type, self.curr_distance[0], self.curr_distance[1], self.curr_violations,
             self.curr_schedule, None])

        # iterate
        for i in range(self.iterations - 1):
            repair_interval = 10000

            if (i + 2) % repair_interval == 0:
                self.row_constraint_repair(self.curr_schedule)
                schedule_difference = self.schedule_absolute_difference(self.schedule_before_repair, self.curr_schedule)
                self.total_schedule_absolute_difference += schedule_difference
            else:
                mutated_schedule, mutated_distance, mutated_violations = self.row_mutation(schedule)
                self.accept(mutated_schedule, mutated_distance, mutated_violations)

            # update temperatue
            self.update_temp(i, self.iterations, cooling_type)

            # iteration before repair
            if (i + 3) % repair_interval == 0:
                self.schedule_before_repair = copy.deepcopy(self.curr_schedule)
                result_writer.writerow(
                    [i + 2, self.n_teams, self.schedule_type, self.curr_distance[0], self.curr_distance[1],
                     self.curr_violations, self.curr_schedule, None])
            # iteration of the repair
            elif (i + 2) % repair_interval == 0:
                result_writer.writerow(
                    [i + 2, self.n_teams, self.schedule_type, self.curr_distance[0], self.curr_distance[1],
                     self.curr_violations, self.curr_schedule, self.nr_schedule_absolute_difference])
            else:
                result_writer.writerow(
                    [i + 2, self.n_teams, self.schedule_type, self.curr_distance[0], self.curr_distance[1],
                     self.curr_violations, None, None])

        # print(self.best_distance[0], self.accepted_mutations, self.violations, self.best_schedule)
        average_schedule_difference = self.total_schedule_absolute_difference/10

        return self.initial_distance, self.best_distance, self.accepted_mutations, self.initial_violations, \
            self.best_violations, self.initial_schedule, self.best_schedule, average_schedule_difference


def run_simulated_annealing_trial(params):
    objective_focus, schedule_type, trial, n_teams, nr_iterations, cooling_type, dataset = params

    print(f"Running Trial {trial} with: {params}")

    directory_path = f"results/deterministic_repair_{nr_iterations}/{str(schedule_type)}/{trial}/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    file_path = directory_path + f"best_{dataset}.txt"
    write_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
    best_file = open(file_path, "a", newline="")  # Keep using append mode
    best_writer = csv.writer(best_file)
    if write_header:
        best_writer.writerow(
            ['n', 'schedule_type', 'init_distance', 'distance', 'init_distance_per_team', 'distance_per_team',
             'accepted_mutations', 'init_violations', 'violations', 'schedule_difference', 'init_schedule',
             'schedule', 'run_time'])

    if dataset == "NLx":
        distance_matrix_path = "datasets/real_life/NL16.txt"
        dataset_name = f"NL{n_teams}"
        distance_matrix = pd.read_csv(distance_matrix_path, header=None, sep=r'\s+').to_numpy(dtype=int)
    elif dataset == "CIRCx":
        distance_matrix_path = f"datasets/real_life/circ{n_teams}.txt"
        dataset_name = f"CIRC{n_teams}"
        distance_matrix = pd.read_csv(distance_matrix_path, header=None, sep=r'\s+').to_numpy(dtype=int)
    else:
        # self-made dataset
        distance_matrix_path = "datasets/SM.csv"
        dataset_name = f"SM{n_teams}"
        distance_matrix = pd.read_csv(distance_matrix_path, header=None).to_numpy(dtype=int)


    result_file = open(directory_path + dataset_name + ".txt", "w")

    start = timeit.default_timer()

    # initialize schedule and Simulated Annealing
    init_schedule = InitSchedule(n_teams=n_teams, schedule_type=schedule_type)
    simulated_annealing = SimulatedAnnealing(init_schedule=init_schedule.get_schedule(),
                                             distance_matrix=distance_matrix,
                                             schedule_type=schedule_type, iterations=nr_iterations,
                                             objective_focus=objective_focus)

    # run SA
    initial_distance, best_distance, accepted_mutations, initial_violations, best_violations, \
        initial_schedule, best_schedule, schedule_difference = simulated_annealing.anneal(result_file, cooling_type=cooling_type)

    stop = timeit.default_timer()
    time = stop - start

    best_writer.writerow(
        [n_teams, schedule_type, initial_distance[0], best_distance[0], initial_distance[1], best_distance[1],
         accepted_mutations,
         initial_violations, best_violations, schedule_difference, initial_schedule, best_schedule, time])

    best_file.close()

    return f"Completed trial {trial} for {objective_focus} - {schedule_type} with n_teams {n_teams}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Simulated Annealing for the TTP.")

    parser.add_argument('--instance', type=str, required=True, help='Dataset instance, e.g., SM12, NL10, CIRC8')
    parser.add_argument('--iterations', type=int, default=100000, help='Number of SA iterations')
    parser.add_argument('--cooling', choices=['geometric', 'linear', 'linear_reheat', 'stairs', 'GG_50'],
                        default='geometric', help='Cooling strategy')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials')

    args = parser.parse_args()

    match = re.match(r"(SM|NL|CIRC)(\d+)", args.instance)
    if not match:
        raise ValueError("Invalid format for --instance. Use format like SM4, NL10, or CIRC20.")

    dataset_prefix, size_str = match.groups()
    n_teams = int(size_str)

    # Map prefix to dataset identifier
    if dataset_prefix == "SM":
        dataset = "SelfMade"
    elif dataset_prefix == "NL":
        dataset = "NLx"
    elif dataset_prefix == "CIRC":
        dataset = "CIRCx"
    else:
        raise ValueError("Dataset must be one of: SM, NL, CIRC")

    print(f"Running {args.trials} trial(s) for {args.instance}:")
    print(f"- Teams: {n_teams}, Iterations: {args.iterations}")
    print(f"- Cooling: {args.cooling}")

    params_list = []
    for trial in range(1, args.trials + 1):
        params_list.append((
            "distance",
            "row",
            trial,
            n_teams,
            args.iterations,
            args.cooling,
            dataset
        ))

    print(f"Running {len(params_list)} trial(s) for {args.instance}")

    # multiprocessing
    with Pool(processes=min(8, args.trials)) as pool:
        results = pool.map(run_simulated_annealing_trial, params_list)

    print("All trials completed.")
