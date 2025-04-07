import random
import pandas as pd
import csv
import os
import timeit
import math
import numpy as np
from multiprocessing import Pool
from collections import defaultdict, Counter
import copy
import argparse
import re


class InitSchedule:
    def __init__(self, n_teams, schedule_type):
        self.n_teams = n_teams
        self.schedule_type = schedule_type
        self.rounds = (n_teams - 1) * 2
        self.column_schedule = self.generate_random_column_first()

    def generate_random_column_first(self):
        schedule_matrix = [[None] * self.n_teams for _ in range(self.rounds)]

        teams = list(range(-self.n_teams, self.n_teams + 1))
        teams.remove(0)

        for team in range(self.n_teams):
            team += 1
            teams_to_pick = teams.copy()
            teams_to_pick.remove(team)  # remove current team
            teams_to_pick.remove(-team)
            for round in range(self.rounds):  # pick opponent for each round
                opponent = random.choice(teams_to_pick)
                teams_to_pick.remove(opponent)

                schedule_matrix[round][team - 1] = opponent

        return schedule_matrix

    def get_schedule(self):
        return self.column_schedule


class SimulatedAnnealing:
    def __init__(self, init_schedule, distance_matrix, schedule_type, temp=1000, alpha=0.9,
                 end_temp=0.00001, iterations=100000):
        self.init_temp = temp
        self.end_temp = end_temp
        self.temp = temp
        self.alpha = alpha
        self.iterations = iterations

        # random schedule
        self.distance_matrix = distance_matrix
        self.schedule_type = schedule_type
        self.n_teams = len(init_schedule[0])  # n_teams
        self.rounds = len(init_schedule)  # rounds

        # inital
        self.initial_schedule = init_schedule
        self.initial_distance = self.calculate_travel_distance(init_schedule, distance_matrix)
        self.initial_violations = self.check_constraints_violated(init_schedule)[0]

        self.best_schedule = init_schedule
        self.best_distance = self.calculate_travel_distance(init_schedule, distance_matrix)
        self.best_violations = self.check_constraints_violated(init_schedule)[0]

        self.curr_schedule = init_schedule
        self.curr_distance = self.calculate_travel_distance(init_schedule, distance_matrix)
        self.curr_violations = self.check_constraints_violated(init_schedule)[0]

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

    def check_constraints_violated(self, schedule):
        max_streak = 3

        violations = {"total": 0, "double_round_robin": 0, "noRepeat": 0, "maxStreak": 0}
        match_count = {}  # track how many times each matchup has occurred
        last_opponent = [-1] * self.n_teams  # store last opponent per team
        home_streaks = [0] * self.n_teams
        away_streaks = [0] * self.n_teams
        last_location = [None] * self.n_teams  # "home" or "away"

        repeat_violations = set()
        streak_violations = set()

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
        round_team_positions = defaultdict(lambda: defaultdict(list))
        bad_matches = []

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
                    bad_matches.append((round_idx, team_idx, abs_opponent, expected_response, actual_response))
                    violations["double_round_robin"] += 1
                    violations["total"] += 1

                round_team_positions[abs_opponent][round_idx].append(team_idx)

                # drr - team plays once per round (column only)
                if abs_opponent in teams_played:
                    # print("played more than once")
                    violations["double_round_robin"] += 1
                    violations["total"] += 1

                teams_played.add(abs_opponent)

                # drr - each matchup occurs twice (row only)
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

                # no Repeat (no consecutive matches against the same team)
                if last_opponent[team_idx] == abs_opponent:
                    violations["noRepeat"] += 1
                    violations["total"] += 1
                    repeat_violations.add(round_idx)

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
                    streak_violations.add(round_idx)

        # ddr: matches that still remain (includes one home, one away)
        remaining_matches = len(required_games)
        violations["double_round_robin"] += remaining_matches
        violations["total"] += remaining_matches

        return violations, round_team_positions, bad_matches, repeat_violations, streak_violations

    def column_mutation(self, schedule):
        '''Swap 2 teams column wise'''
        # swap team column
        # randomly choose team idx
        rand_team = random.randrange(self.n_teams)

        # randomly choose 2 rounds
        indices = random.sample(range(self.rounds), 2)

        rand_team1_element = schedule[indices[0]][rand_team]
        rand_team2_element = schedule[indices[1]][rand_team]

        schedule[indices[0]][rand_team] = rand_team2_element
        schedule[indices[1]][rand_team] = rand_team1_element

        mutated_distance = self.calculate_travel_distance(schedule, self.distance_matrix)
        mutated_violations, *_ = self.check_constraints_violated(schedule)

        return schedule, mutated_distance, mutated_violations

    def check_inconsistent_matches(self, schedule):
        inconsistencies = []
        for round_idx, round_matches in enumerate(schedule):
            for team_idx in range(self.n_teams):
                opponent = round_matches[team_idx]
                abs_opponent = abs(opponent) - 1
                expected = -(team_idx + 1) if opponent > 0 else team_idx + 1
                actual = schedule[round_idx][abs_opponent]
                if actual != expected:
                    inconsistencies.append((round_idx, team_idx, abs_opponent, expected, opponent))
        return inconsistencies

    def repair_inconsistent_matches(self, schedule, bad_matches, prev_swaps=None):
        schedule = [row[:] for row in schedule]  # Deep copy
        repaired_line = []
        swaps_done = set()
        teams_repaired = set()

        if prev_swaps is None:
            prev_swaps = set()

        updated_matches = list(bad_matches)
        random.shuffle(updated_matches)
        i = 0
        while i < len(updated_matches):
            r1, t1, o1, e1, f1 = updated_matches[i]
            pos1 = (r1, o1)
            if pos1 in swaps_done:
                i += 1
                continue

            pair_key = (r1, tuple(sorted([t1, o1])))
            if pair_key in teams_repaired:
                i += 1
                continue

            for j in range(i + 1, len(updated_matches)):
                r2, t2, o2, e2, f2 = updated_matches[j]
                pos2 = (r2, o2)
                if o1 != o2 or pos2 in swaps_done:
                    continue

                swap_key = tuple(sorted([(r1, o1), (r2, o2)]))
                if swap_key in prev_swaps:
                    continue

                if e1 == f2 and e2 == f1 or set([t1, o1]) == set([t2, o2]):
                    schedule[r1][o1], schedule[r2][o2] = schedule[r2][o2], schedule[r1][o1]
                    repaired_line.append(
                        f"Round {r1}: Team {t1 + 1} vs {o1 + 1} — Expected: {e1}, Found: {f1} - Swap {r2}, {o2}")
                    swaps_done.update([pos1, pos2])
                    teams_repaired.update([pair_key, (r2, tuple(sorted([t2, o2])))])
                    prev_swaps.add(swap_key)
                    updated_matches.pop(j)
                    updated_matches.pop(i)
                    break
            else:
                i += 1

        new_bad_matches = self.check_inconsistent_matches(schedule)
        return schedule, new_bad_matches, repaired_line, prev_swaps

    def smart_round_swaps(self, schedule, violation_rounds, violation_type="streak"):
        round_counts = Counter(violation_rounds)
        ordered_rounds = [r for r, _ in round_counts.most_common()]
        random.shuffle(ordered_rounds)

        for i in range(0, len(ordered_rounds) - 1, 2):
            r1, r2 = ordered_rounds[i], ordered_rounds[i + 1]
            temp_schedule = copy.deepcopy(schedule)
            temp_schedule[r1], temp_schedule[r2] = temp_schedule[r2], temp_schedule[r1]

            _, _, _, repeat_viol_temp, streak_viol_temp = self.check_constraints_violated(temp_schedule)

            if violation_type == "streak":
                if len(streak_viol_temp) < len(violation_rounds):
                    schedule[r1], schedule[r2] = schedule[r2], schedule[r1]
            elif violation_type == "repeat":
                if len(repeat_viol_temp) < len(violation_rounds):
                    schedule[r1], schedule[r2] = schedule[r2], schedule[r1]

        return schedule

    def apply_drr_repair(self, schedule, round_team_positions):

        for team, round_pos_map in round_team_positions.items():
            for round_idx, positions in round_pos_map.items():
                if len(positions) <= 1:
                    continue

                for pos in positions:
                    for other_team, other_round_pos_map in round_team_positions.items():
                        if team == other_team:
                            continue

                        for other_round_idx, other_positions in other_round_pos_map.items():
                            if round_idx == other_round_idx or pos not in other_positions:
                                continue

                            team_opponents = [abs(schedule[other_round_idx][c]) - 1 for c in range(self.n_teams)]
                            other_team_opponents = [abs(schedule[round_idx][c]) - 1 for c in range(self.n_teams)]

                            if team in team_opponents or other_team in other_team_opponents:
                                continue

                            schedule[round_idx][pos], schedule[other_round_idx][pos] = (
                                schedule[other_round_idx][pos],
                                schedule[round_idx][pos],
                            )

        # handle noRepeat and maxStreak violations with smarter round swaps
        violations, _, _, repeat_violations, streak_violations = self.check_constraints_violated(schedule)

        schedule = self.smart_round_swaps(schedule, list(streak_violations), violation_type="streak")

        # recheck repeat violations after streak fix
        _, _, _, repeat_violations, _ = self.check_constraints_violated(schedule)
        schedule = self.smart_round_swaps(schedule, list(repeat_violations), violation_type="repeat")

        return schedule

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
                self.best_schedule, self.best_distance, self.best_violations = mutated_schedule, mutated_distance, mutated_violations

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
            self.temp = self.init_temp * (1 - (itr_nr / max_itr))

    def schedule_absolute_difference(self, schedule1, schedule2):
        arr1 = np.array(schedule1)
        arr2 = np.array(schedule2)
        difference = np.sum(arr1 != arr2)
        self.nr_schedule_absolute_difference = difference
        return difference

    def anneal(self, result_file, cooling_type):
        # rand schedule + distance
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
                violations, round_team_positions, bad_matches, _, _ = self.check_constraints_violated(schedule)

                self.curr_schedule = self.apply_drr_repair(self.curr_schedule, round_team_positions)
                self.curr_distance = self.calculate_travel_distance(self.curr_schedule, self.distance_matrix)
                self.curr_violations, *_ = self.check_constraints_violated(self.curr_schedule)[0]

                schedule_difference = self.schedule_absolute_difference(self.schedule_before_repair, self.curr_schedule)
                self.total_schedule_absolute_difference += schedule_difference

            else:
                mutated_schedule, mutated_distance, mutated_violations = self.column_mutation(schedule)
                self.accept(mutated_schedule, mutated_distance, mutated_violations)

            # update temperatue
            self.update_temp(i, self.iterations, cooling_type)

            if (i + 3) % (repair_interval) == 0:
                result_writer.writerow(
                    [i + 2, self.n_teams, self.schedule_type, self.curr_distance[0], self.curr_distance[1],
                     self.curr_violations, self.curr_schedule, None])
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
    schedule_type, trial, n_teams, nr_iterations, cooling_type, dataset = params

    # Print to debug and ensure unique parameters
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
             'accepted_mutations', 'init_violations', 'violations', 'init_schedule', 'schedule', 'run_time'])

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

    # Initialize schedule and Simulated Annealing
    init_schedule = InitSchedule(n_teams=n_teams, schedule_type=schedule_type)
    simulated_annealing = SimulatedAnnealing(init_schedule=init_schedule.get_schedule(),
                                             distance_matrix=distance_matrix,
                                             schedule_type=schedule_type, iterations=nr_iterations)

    # run SA
    # initial_distance, best_distance, accepted_mutations, initial_violations, best_violations, \
    #     initial_schedule, best_schedule = simulated_annealing.anneal(result_file, cooling_type=cooling_type)

    initial_distance, best_distance, accepted_mutations, initial_violations, best_violations, \
        initial_schedule, best_schedule, schedule_difference = simulated_annealing.anneal(result_file,
                                                                                          cooling_type=cooling_type)

    stop = timeit.default_timer()
    time = stop - start

    # best_writer.writerow(
    #     [n_teams, schedule_type, initial_distance[0], best_distance[0], initial_distance[1], best_distance[1],
    #      accepted_mutations,
    #      initial_violations, best_violations, initial_schedule, best_schedule, time])

    best_writer.writerow(
        [n_teams, schedule_type, initial_distance[0], best_distance[0], initial_distance[1], best_distance[1],
         accepted_mutations,
         initial_violations, best_violations, schedule_difference, initial_schedule, best_schedule, time])

    best_file.close()

    return f"Completed trial {trial} - {schedule_type} with n_teams {n_teams}"


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
            "column",
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
