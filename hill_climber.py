import random
import pandas as pd
import csv
import os
import timeit
import copy
from multiprocessing import Pool
import argparse
import re


class InitSchedule:
    def __init__(self, n_teams, schedule_type):
        self.n_teams = n_teams
        self.rounds = (n_teams - 1) * 2
        self.row_schedule = self.generate_random_row_first()
        self.column_schedule = self.generate_random_column_first()

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

    def get_schedule(self, schedule_type):
        if schedule_type == "row":
            return self.row_schedule
        else:
            return self.column_schedule


class HillClimbing:
    def __init__(self, init_schedule, distance_matrix, schedule_type, iterations):
        self.iterations = iterations

        # random schedule
        self.best_schedule = init_schedule
        self.distance_matrix = distance_matrix
        self.schedule_type = schedule_type
        self.n_teams = len(init_schedule[0])  # n_teams
        self.rounds = len(init_schedule)  # rounds

        self.inital_distance = self.calculate_travel_distance(init_schedule, distance_matrix)
        self.inital_violations = self.check_constraints_violated(init_schedule)

        self.best_distance = self.calculate_travel_distance(init_schedule, distance_matrix)
        self.violations = self.check_constraints_violated(init_schedule)

        self.accepted_mutations = 0
        self.improve_type = 'none'

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

            # return to home if last game was away
            if prev_location == "away":
                return_home_distance = distance_matrix[prev_opponent, team_idx]
                total_distance += return_home_distance
                dist_per_team[team_idx] += return_home_distance

        return total_distance, [int(d) for d in dist_per_team]

    def check_constraints_violated(self, schedule):
        max_streak = 3

        violations = {"total": 0, "double_round_robin": 0, "noRepeat": 0, "maxStreak": 0}
        match_count = {}  # track how many times each matchup has occurred
        last_opponent = [-1] * self.n_teams  # store last opponent per team
        home_streaks = [0] * self.n_teams
        away_streaks = [0] * self.n_teams
        last_location = [None] * self.n_teams  # "home" or "away"

        required_matches = set()
        for team_idx in range(self.n_teams):
            for opponent_idx in range(team_idx + 1, self.n_teams):  # Only generate unique pairs
                required_matches.add(frozenset([team_idx, opponent_idx]))  # Team X vs Team Y (unordered)

        # to generate the home and away matches, need (team_idx, opponent_idx) and (opponent_idx, team_idx)
        match_pairs = []
        for match in required_matches:
            team1, team2 = list(match)
            # add (team1 vs team2) and (team2 vs team1)
            match_pairs.append((team1, team2))
            match_pairs.append((team2, team1))

        required_games = set(match_pairs)
        matches_more_than_twice = []

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

                # drr - team plays once per round
                if abs_opponent in teams_played:
                    violations["double_round_robin"] += 1
                    violations["total"] += 1

                teams_played.add(abs_opponent)

                # double round-robin constraint (each matchup occurs twice)
                match = tuple([abs_opponent, team_idx])
                match_count[match] = match_count.get(match, 0) + 1
                # ff a match occurs more than twice, it's a violation
                if match_count[match] > 2:
                    if match not in matches_more_than_twice:
                        # print("match more than twice")
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

        return violations

    def row_mutation(self, schedule):
        '''Swap 2 teams row wise'''
        # swap team row
        # randomly choose round
        rand_round = random.choice([0, self.rounds - 1])

        # randomly choose 2 teams
        indices = random.sample(range(self.n_teams), 2)
        rand_team1_element = schedule[rand_round][indices[0]]
        rand_team2_element = schedule[rand_round][indices[1]]

        # check if the team playing itself
        while abs(rand_team1_element) == indices[1] + 1 or abs(rand_team2_element) == indices[0] + 1:
            indices = random.sample(range(self.n_teams), 2)
            rand_team1_element = schedule[rand_round][indices[0]]
            rand_team2_element = schedule[rand_round][indices[1]]

        # swap the teams
        schedule[rand_round][indices[0]] = rand_team2_element
        schedule[rand_round][indices[1]] = rand_team1_element

        return schedule

    def column_mutation(self, schedule):
        '''Swap 2 teams column wise'''
        new_schedule = copy.deepcopy(schedule)
        # randomly choose team idx
        rand_team = random.randrange(self.n_teams)

        # randomly choose 2 rounds
        indices = random.sample(range(self.rounds), 2)

        rand_team1_element = new_schedule[indices[0]][rand_team]
        rand_team2_element = new_schedule[indices[1]][rand_team]

        new_schedule[indices[0]][rand_team] = rand_team2_element
        new_schedule[indices[1]][rand_team] = rand_team1_element

        return new_schedule

    # hill climbing
    def climbing(self, result_file, objective_focus):
        # rand schedule + distance
        schedule = self.best_schedule

        result_writer = csv.writer(result_file)
        result_writer.writerow(
            ['iteration', 'n', 'schedule_type', 'distance', 'distance_per_team', 'violations', 'schedule', 'improvement type'])
        result_writer.writerow(
            [1, self.n_teams, self.schedule_type, self.best_distance[0], self.best_distance[1], self.violations,
             self.best_schedule, self.improve_type])

        for i in range(self.iterations-1):
            # apply mutation
            if self.schedule_type == "row":
                mutated_schedule = self.row_mutation(schedule)
            else:
                mutated_schedule = self.column_mutation(schedule)

            mutated_distance = self.calculate_travel_distance(mutated_schedule, self.distance_matrix)
            mutated_violations = self.check_constraints_violated(mutated_schedule)

            if objective_focus == "distance":
                # compare old and new dsitance
                if mutated_distance <= self.best_distance:
                    self.accepted_mutations += 1
                    self.best_schedule = mutated_schedule  # new schedule
                    self.best_distance = mutated_distance
                    self.violations = mutated_violations

            elif objective_focus == "violations":
                if mutated_violations['total'] <= self.violations['total']:
                    self.accepted_mutations += 1
                    self.best_schedule = mutated_schedule  # new schedule
                    self.best_distance = mutated_distance
                    self.violations = mutated_violations
            else:
                if mutated_violations['total'] <= self.violations['total'] or mutated_distance <= self.best_distance:
                    if mutated_distance <= self.best_distance:
                        self.improve_type = "distance"
                    else:
                        self.improve_type = "violations"
                        self.accepted_mutations += 1
                    self.best_schedule = mutated_schedule  # new schedule
                    self.best_distance = mutated_distance
                    self.violations = mutated_violations

            if (i + 2) % 10000 == 0:
                result_writer.writerow(
                    [i+2, self.n_teams, self.schedule_type, self.best_distance[0], self.best_distance[1], self.violations, self.best_schedule])
            else:
                result_writer.writerow(
                    [i+2, self.n_teams, self.schedule_type, self.best_distance[0], self.best_distance[1],self.violations, None])

        return self.inital_distance, self.best_distance, self.accepted_mutations, self.inital_violations, self.violations, self.best_schedule


# execute for each trial and schedule size
def run_hill_climbing(params):
    objective_focus, schedule_type, trial, n_teams, nr_iterations, dataset = params

    print(f"Running Trial {trial} with: {params}")

    directory_path = f"results/HC_{nr_iterations}/{objective_focus}/{str(schedule_type)}/{trial}/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

    best_file = open(directory_path + f"best_{dataset}.txt", "a")
    write_header = not os.path.exists(directory_path + f"best_{dataset}.txt") or os.stat(directory_path + f"best_{dataset}.txt").st_size == 0
    best_writer = csv.writer(best_file)
    if write_header:
        best_writer.writerow(
            ['n', 'schedule_type', 'initial_distance','distance', 'distance_per_team', 'accepted_mutations',
             'initial_violations','violations', 'schedule', 'run_time']
        )

    # distance matrix
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

    init_schedule = InitSchedule(n_teams=n_teams, schedule_type=schedule_type)

    # run the climbing algorithm
    hill_climber = HillClimbing(init_schedule=init_schedule.get_schedule(schedule_type), distance_matrix=distance_matrix,
                                schedule_type=schedule_type, iterations=nr_iterations)

    inital_distance, best_distance, accepted_mutations, initial_violations, violations, best_schedule = hill_climber.climbing(result_file, objective_focus)

    stop = timeit.default_timer()
    time = stop - start
    best_writer.writerow(
        [n_teams, schedule_type, inital_distance[0 ], best_distance[0], best_distance[1], accepted_mutations,
         initial_violations, violations, best_schedule,  time])

    return f"Completed trial {trial} for {objective_focus} - {schedule_type} with n_teams {n_teams}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run hill climbing for one dataset instance.")
    parser.add_argument("--instance", type=str, required=True,
                        help="Dataset instance in the format SM4, NL10, or CIRC20")
    parser.add_argument("--iterations", type=int, default=100000, help="Number of hill climbing iterations")
    parser.add_argument("--schedule_type", type=str, choices=["row", "column"], default="column",
                        help="Schedule mutation type")
    parser.add_argument("--objective", type=str, choices=["distance", "violations"], default="distance",
                        help="Objective to optimize")
    parser.add_argument("--trials", type=int, default=1, help="Number of independent HC trials to run")

    args = parser.parse_args()

    match = re.match(r"(SM|NL|CIRC)(\d+)", args.instance)
    if not match:
        raise ValueError("Invalid format for --instance. Use format like SM4, NL10, or CIRC20.")

    dataset_prefix, size_str = match.groups()
    n_teams = int(size_str)

    # map prefix to dataset identifier
    if dataset_prefix == "SM":
        dataset = "SelfMade"
    elif dataset_prefix == "NL":
        dataset = "NLx"
    elif dataset_prefix == "CIRC":
        dataset = "CIRCx"
    else:
        raise ValueError("Dataset must be one of: SM, NL, CIRC")

    print(f"Running {args.trials} hill climbing trial(s) for {args.instance}:")
    print(f"- Teams: {n_teams}, Iterations: {args.iterations}, Schedule: {args.schedule_type}")
    print(f"- Objective: {args.objective}")

    params_list = []
    for trial in range(1, args.trials + 1):
        params_list.append((
            args.objective,
            args.schedule_type,
            trial,  # trial index
            n_teams,
            args.iterations,
            dataset
        ))

    print(f"Running {len(params_list)} trial(s) for {args.instance}")

    # multiprocessing
    with Pool(processes=min(8, args.trials)) as pool:
        results = pool.map(run_hill_climbing, params_list)

    print("All trials completed.")

