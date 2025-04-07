import random
import pandas as pd
import csv
import os
import timeit
import math
import numpy as np
import copy
from multiprocessing import Pool
from ortools.sat.python import cp_model
import argparse


class InitSchedule:
    def __init__(self, n_teams, schedule_type):
        self.n_teams = n_teams
        self.schedule_type = schedule_type
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

    def get_schedule(self):
        if self.schedule_type == "row":
            return self.row_schedule
        else:
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
        self.initial_violations = self.check_constraints_violated(init_schedule)

        self.best_schedule = init_schedule
        self.best_distance = self.calculate_travel_distance(init_schedule, distance_matrix)
        self.best_violations = self.check_constraints_violated(init_schedule)

        self.curr_schedule = init_schedule
        self.curr_distance = self.calculate_travel_distance(init_schedule, distance_matrix)
        self.curr_violations = self.check_constraints_violated(init_schedule)

        self.accepted_mutations = 0
        self.feasible_repairs = 0
        self.nr_schedule_absolute_difference = 0
        self.total_schedule_absolute_difference = 0

        self.zero_vio_schedule = init_schedule
        self.zero_vio_distance = self.calculate_travel_distance(init_schedule, distance_matrix)
        self.zero_vio_violations = self.check_constraints_violated(init_schedule)

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
            for opponent_idx in range(team_idx + 1, self.n_teams):  # only unique pairs
                required_matches.add(frozenset([team_idx, opponent_idx]))  # Team X vs Team Y (unordered)

        # to generate the home and away matches, we both (team_idx, opponent_idx) and (opponent_idx, team_idx)
        match_pairs = []
        for match in required_matches:
            team1, team2 = list(match)
            # add (team1 vs team2) and (team2 vs team1)
            match_pairs.append((team1, team2))
            match_pairs.append((team2, team1))

        required_games = set(match_pairs)
        matches_more_than_twice = []
        inconsistent_matches = []

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
                    inconsistent_matches.append((round_idx, team_idx, abs_opponent, expected_response, actual_response))
                    violations["double_round_robin"] += 1
                    violations["total"] += 1

                # drr - team plays once per round (column only)
                if abs_opponent in teams_played:
                    # print("played more than once")
                    violations["double_round_robin"] += 1
                    violations["total"] += 1

                teams_played.add(abs_opponent)

                # drr - each matchup occurs twice (row only)
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

        # ddr: matches that still remain (includes one home, one away)
        remaining_matches = len(required_games)
        violations["double_round_robin"] += remaining_matches
        violations["total"] += remaining_matches

        return violations

    def strict_cp_sat(self, schedule):
        model = cp_model.CpModel()

        # variables: X[r][i][j] = 1 if team i plays home vs team j in round r
        X = {}
        for r in range(self.rounds):
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i != j:
                        X[r, i, j] = model.NewBoolVar(f"X_r{r}_i{i}_j{j}")

        violations = self.check_constraints_violated(schedule)

        # list of constraints with non-zero violations
        constraints = ["double_round_robin", "noRepeat", "maxStreak"]
        valid_constraints = [c for c in constraints if violations.get(c, 0) > 0]

        # all violations are zero, return the schedule as-is
        if not valid_constraints:
            return schedule, self.curr_schedule, violations, "No valid constraint"

        focus_constraint = "All"

        # DRR
        for r in range(self.rounds):
            for t in range(self.n_teams):
                # Sum of home matches and away matches per round should be exactly 1
                model.Add(
                    sum(X[r, t, o] for o in range(self.n_teams) if t != o) +
                    sum(X[r, o, t] for o in range(self.n_teams) if t != o) == 1
                )

        # requirement 2: Each pair plays exactly twice (home and away)
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i != j:
                    model.Add(sum(X[r, i, j] for r in range(self.rounds)) == 1)

        # No Repeat Constraint
        for t in range(self.n_teams):
            for r in range(self.rounds - 1):
                for o in range(self.n_teams):
                    if t != o:
                        # No repeat in consecutive rounds regardless of home/away
                        model.Add(X[r, t, o] + X[r + 1, t, o] + X[r, o, t] + X[r + 1, o, t] <= 1)

        max_streak = 3
        # Each team plays at most 'max_streak' consecutive home or away games
        for t in range(self.n_teams):
            for r in range(self.rounds - max_streak):
                model.Add(
                    sum(X[r + k, t, o] for k in range(max_streak + 1) for o in range(self.n_teams) if t != o) <= max_streak)
                model.Add(
                    sum(X[r + k, o, t] for k in range(max_streak + 1) for o in range(self.n_teams) if t != o) <= max_streak)

        # hard-code consistent existing valid matches
        for r in range(self.rounds):
            for i in range(self.n_teams):
                opponent = schedule[r][i]
                j = abs(opponent) - 1
                if j >= 0 and j != i:
                    if (opponent > 0 and schedule[r][j] == -(i + 1)):
                        model.Add(X[r, i, j] == 1)
                    elif (opponent < 0 and schedule[r][j] == (i + 1)):
                        model.Add(X[r, j, i] == 1)

        # solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0
        status = solver.Solve(model)

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            distance = self.calculate_travel_distance(schedule, self.distance_matrix)
            violations = self.check_constraints_violated(schedule)

            return schedule, distance, violations, "Not feasible"

        # build repaired schedule
        new_schedule = [row[:] for row in schedule]
        for r in range(self.rounds):
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i != j and solver.Value(X[r, i, j]) == 1:
                        new_schedule[r][i] = j + 1
                        new_schedule[r][j] = -(i + 1)

        distance = self.calculate_travel_distance(new_schedule, self.distance_matrix)
        violations = self.check_constraints_violated(new_schedule)

        self.feasible_repairs += 1

        return new_schedule, distance, violations, "Feasible"

    def soft_cp_sat(self, schedule):
        model = cp_model.CpModel()

        # variables: X[r][i][j] = 1 if team i plays home vs team j in round r
        X = {}
        for r in range(self.rounds):
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i != j:
                        X[r, i, j] = model.NewBoolVar(f"X_r{r}_i{i}_j{j}")

        # Penalty variables
        penalties = []

        violations = self.check_constraints_violated(schedule)

        # list of constraints with non-zero violations
        constraints = ["double_round_robin", "noRepeat", "maxStreak"]
        valid_constraints = [c for c in constraints if violations.get(c, 0) > 0]

        # all violations are zero, return the schedule as-is
        if not valid_constraints:
            return schedule, self.curr_distance, violations, "No valid constraint"

        focus_constraint = "All"

        # DRR
        if focus_constraint == "double_round_robin" or focus_constraint == "All":
            # requirement 1: Each team plays exactly one match per round
            for r in range(self.rounds):
                for t in range(self.n_teams):
                    # Sum of home matches and away matches per round should be exactly 1
                    model.Add(
                        sum(X[r, t, o] for o in range(self.n_teams) if t != o) +
                        sum(X[r, o, t] for o in range(self.n_teams) if t != o) == 1
                    )

            # requirement 2: Each pair plays exactly twice (home and away)
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i != j:
                        # Soft constraint: Minimize the violation if the pair doesn't play exactly twice (once home, once away)
                        drr_penalty = model.NewIntVar(0, 2, f"drr_penalty_{i}_{j}")
                        model.Add(sum(X[r, i, j] for r in range(self.rounds)) == 1 + drr_penalty
                        )
                        penalties.append(drr_penalty)

        # No Repeat
        if focus_constraint == "noRepeat" or focus_constraint == "All":
            for t in range(self.n_teams):
                for r in range(self.rounds - 1):
                    for o in range(self.n_teams):
                        if t != o:
                            # Add penalty for repeat matches
                            repeat_penalty = model.NewIntVar(0, 1, f"repeat_penalty_{r}_{t}_{o}")
                            model.Add(X[r, t, o] + X[r + 1, t, o] + X[r, o, t] + X[r + 1, o, t] <= 1 + repeat_penalty)
                            penalties.append(repeat_penalty)

        # Max Streak
        if focus_constraint == "maxStreak" or focus_constraint == "All":
            max_streak = 3
            for t in range(self.n_teams):
                for r in range(self.rounds - max_streak):
                    home_streak_penalty = model.NewIntVar(0, 1, f"home_streak_penalty_{r}_{t}")
                    away_streak_penalty = model.NewIntVar(0, 1, f"away_streak_penalty_{r}_{t}")
                    model.Add(
                        sum(X[r + k, t, o] for k in range(max_streak + 1) for o in range(self.n_teams) if
                            t != o) <= max_streak + home_streak_penalty
                    )
                    model.Add(
                        sum(X[r + k, o, t] for k in range(max_streak + 1) for o in range(self.n_teams) if
                            t != o) <= max_streak + away_streak_penalty
                    )
                    penalties.extend([home_streak_penalty, away_streak_penalty])

        # minimize the sum of penalties
        model.Minimize(sum(penalties))

        # hard-code consistent existing valid matches
        for r in range(self.rounds):
            for i in range(self.n_teams):
                opponent = schedule[r][i]
                j = abs(opponent) - 1
                if j >= 0 and j != i:
                    if (opponent > 0 and schedule[r][j] == -(i + 1)):
                        model.Add(X[r, i, j] == 1)
                    elif (opponent < 0 and schedule[r][j] == (i + 1)):
                        model.Add(X[r, j, i] == 1)

        # solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0
        status = solver.Solve(model)

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            distance = self.calculate_travel_distance(schedule, self.distance_matrix)
            violations = self.check_constraints_violated(schedule)

            return schedule, distance, violations, "Not feasible"

        # build repaired schedule
        new_schedule = [row[:] for row in schedule]
        for r in range(self.rounds):
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i != j and solver.Value(X[r, i, j]) == 1:
                        new_schedule[r][i] = j + 1
                        new_schedule[r][j] = -(i + 1)

        distance = self.calculate_travel_distance(new_schedule, self.distance_matrix)
        violations = self.check_constraints_violated(new_schedule)

        self.feasible_repairs += 1

        return new_schedule, distance, violations, "Feasible"

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
        mutated_violations = self.check_constraints_violated(new_schedule)

        return schedule, mutated_distance, mutated_violations

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
        mutated_violations = self.check_constraints_violated(schedule)

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
        # compare old and new dsitance
        if mutated_distance <= self.curr_distance:
            self.curr_schedule, self.curr_distance, self.curr_violations = mutated_schedule, mutated_distance, mutated_violations
            self.accepted_mutations += 1
            if mutated_distance < self.best_distance:
                self.best_schedule, self.best_distance, self.best_violations = mutated_schedule, mutated_distance, mutated_violations

        else:
            if random.random() < self.acceptance_prob(mutated_distance, self.curr_distance):
                self.curr_schedule, self.curr_distance, self.curr_violations = mutated_schedule, mutated_distance, mutated_violations

        if self.curr_violations["total"] == 0 and self.curr_distance[0] < self.zero_vio_distance[0]:
            self.zero_vio_schedule, self.zero_vio_distance, self.zero_vio_violations = mutated_schedule, mutated_distance, mutated_violations

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

    def check_best_schedule(self, new_schedule, distance, violations):
        self.curr_schedule, self.curr_distance, self.curr_violations = new_schedule, distance, violations

        if self.curr_distance < self.best_distance:
            self.best_schedule, self.best_distance, self.best_violations = new_schedule, distance, violations

        if self.curr_violations["double_round_robin"] == 0:
            self.fixed_rows = True

        if self.curr_violations["total"] == 0 and self.curr_distance[0] < self.zero_vio_distance[0]:
            self.zero_vio_schedule, self.zero_vio_distance, self.zero_vio_violations = new_schedule, distance, violations

    def anneal(self, result_file, cooling_type, CP_SAT_type):
        # rand schedule + distance
        schedule = self.best_schedule

        result_writer = csv.writer(result_file)
        result_writer.writerow(
            ['iteration', 'n', 'schedule_type', 'distance', 'distance_per_team', 'violations', 'schedule', 'fixed_cosntraints', 'schedule_absolute_diff'])
        result_writer.writerow(
            [1, self.n_teams, self.schedule_type, self.curr_distance[0], self.curr_distance[1], self.curr_violations,
             self.curr_schedule, None, None])

        # iterate
        for i in range(self.iterations - 1):

            repair_interval = 10000

            # appy CP-SAT
            if (i + 2) % repair_interval == 0:
                if CP_SAT_type == "strict":
                    new_schedule, distance, violations, fix = self.strict_cp_sat(self.curr_schedule)
                    schedule_difference = self.schedule_absolute_difference(self.schedule_before_repair,
                                                                            self.curr_schedule)
                    self.total_schedule_absolute_difference += schedule_difference
                    self.check_best_schedule(new_schedule, distance, violations)
                elif CP_SAT_type == "soft":
                    new_schedule, distance, violations, fix = self.soft_cp_sat(self.curr_schedule)
                    self.check_best_schedule(new_schedule, distance, violations)

            else:
                # apply mutation
                if self.schedule_type == "row":
                    mutated_schedule, mutated_distance, mutated_violations = self.row_mutation(self.curr_schedule)
                else:
                    mutated_schedule, mutated_distance, mutated_violations = self.column_mutation(self.curr_schedule)

                # acceptance
                self.accept(mutated_schedule, mutated_distance, mutated_violations)

            # update temperatue
            self.update_temp(i, self.iterations, cooling_type)

            if (i + 3) % repair_interval == 0:
                self.schedule_before_repair = copy.deepcopy(self.curr_schedule)
                result_writer.writerow(
                    [i + 2, self.n_teams, self.schedule_type, self.curr_distance[0], self.curr_distance[1],
                     self.curr_violations, self.curr_schedule, None, None])
            elif (i + 2) % repair_interval == 0:
                result_writer.writerow(
                    [i + 2, self.n_teams, self.schedule_type, self.curr_distance[0], self.curr_distance[1],
                     self.curr_violations, self.curr_schedule, fix, self.nr_schedule_absolute_difference])
            else:
                result_writer.writerow(
                    [i + 2, self.n_teams, self.schedule_type, self.curr_distance[0], self.curr_distance[1],
                     self.curr_violations, None, None, None])

        # print(self.best_distance[0], self.accepted_mutations, self.violations, self.best_schedule)
        average_schedule_difference = self.total_schedule_absolute_difference/10

        return self.initial_distance, self.best_distance, self.accepted_mutations, self.initial_violations, \
            self.best_violations, self.initial_schedule, self.best_schedule, self.zero_vio_schedule, \
            self.zero_vio_distance, self.zero_vio_violations, self.feasible_repairs, average_schedule_difference


def run_simulated_annealing_trial(params):
    schedule_type, trial, n_teams, nr_iterations, cooling_type, dataset, CP_SAT_type = params

    print(f"Running Trial {trial} with: {params}")

    if CP_SAT_type == "strict":
        directory_path = f"results/CP_SAT_Strict_{nr_iterations}/{cooling_type}/{str(schedule_type)}/{trial}/"
    elif CP_SAT_type == "soft":
        directory_path = f"results/CP_SAT_Soft_{nr_iterations}/{cooling_type}/{str(schedule_type)}/{trial}/"

    os.makedirs(directory_path, exist_ok=True)

    best_distance_file = open(directory_path + f"best_distance_{dataset}.txt", "a")  # Change 'w' to 'a' for appending
    dist_write_header = not os.path.exists(directory_path + f"best_distance_{dataset}.txt") or os.stat(directory_path + f"best_distance_{dataset}.txt").st_size == 0
    best_distance_writer = csv.writer(best_distance_file)
    if dist_write_header:
        best_distance_writer.writerow(
            ['n', 'schedule_type', 'init_distance', 'distance', 'init_distance_per_team', 'distance_per_team',
             'accepted_mutations', 'init_violations', 'violations', 'feasible_repairs', 'schedule_difference', 'init_schedule', 'schedule', 'run_time'])

    best_violations_file = open(directory_path + f"best_violations_{dataset}.txt", "a")  # Change 'w' to 'a' for appending
    viol_write_header = not os.path.exists(directory_path + f"best_violations_{dataset}.txt") or os.stat(directory_path + f"best_violations_{dataset}.txt").st_size == 0
    best_violations_writer = csv.writer(best_violations_file)
    if viol_write_header:
        best_violations_writer.writerow(
            ['n', 'schedule_type', 'init_distance', 'distance', 'init_distance_per_team', 'distance_per_team',
             'accepted_mutations', 'init_violations', 'violations', 'feasible_repairs', 'schedule_difference', 'init_schedule', 'schedule', 'run_time'])

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

    # init schedule and Simulated Annealing
    init_schedule = InitSchedule(n_teams=n_teams, schedule_type=schedule_type)
    simulated_annealing = SimulatedAnnealing(init_schedule=init_schedule.get_schedule(),
                                             distance_matrix=distance_matrix,
                                             schedule_type=schedule_type, iterations=nr_iterations)

    # run SA
    initial_distance, best_distance, accepted_mutations, initial_violations, best_violations, \
        initial_schedule, best_schedule, zero_vio_schedule, zero_vio_distance, \
        zero_vio_violations, feasible_repairs, schedule_difference = simulated_annealing.anneal(result_file, cooling_type, CP_SAT_type)

    stop = timeit.default_timer()
    time = stop - start

    best_distance_writer.writerow(
        [n_teams, schedule_type, initial_distance[0], best_distance[0], initial_distance[1], best_distance[1],
         accepted_mutations, initial_violations, best_violations, feasible_repairs, schedule_difference, initial_schedule, best_schedule, time])

    best_violations_writer.writerow(
        [n_teams, schedule_type, initial_distance[0], zero_vio_distance[0], initial_distance[1], zero_vio_distance[1],
         accepted_mutations, initial_violations, zero_vio_violations, feasible_repairs, schedule_difference, initial_schedule, zero_vio_schedule, time])

    return f"Completed trial {trial} - {schedule_type} with n_teams {n_teams}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulated Annealing with CP-SAT repair')

    parser.add_argument('--instance', type=str, required=True, help='Instance name, e.g., SM12, CIRC20, NL16')
    parser.add_argument('--iterations', type=int, default=100000, help='Number of iterations')
    parser.add_argument('--schedule_type', type=str, choices=['row', 'column'], default='column', help='Mutation schedule type')
    parser.add_argument(
        '--cooling',
        type=str,
        default='geometric',
        choices=['geometric', 'linear', 'linear_reheat', 'stairs', 'GG_50'],
        help='Cooling schedule type'
    )
    parser.add_argument('--trials', type=int, default=1, help='Number of trials to run')
    parser.add_argument('--cpsat_mode', type=str, choices=['strict', 'soft'], default='strict', help='CP-SAT repair mode')

    args = parser.parse_args()

    # Extract dataset type and team count
    if args.instance.startswith("SM"):
        dataset = "SelfMade"
        n_teams = int(args.instance[2:])
    elif args.instance.startswith("CIRC"):
        dataset = "CIRCx"
        n_teams = int(args.instance[4:])
    elif args.instance.startswith("NL"):
        dataset = "NLx"
        n_teams = int(args.instance[2:])
    else:
        raise ValueError("Unsupported dataset instance format. Use SM12, NL16, or CIRC20.")

    # parameters list
    params_list = []
    for trial in range(1, args.trials + 1):
        params_list.append((
            args.schedule_type,
            trial,
            n_teams,
            args.iterations,
            args.cooling,
            dataset,
            args.cpsat_mode
        ))

    print(f"Running {len(params_list)} trial(s) for {args.instance} using CP-SAT {args.cpsat_mode} mode")

    # Use multiprocessing
    with Pool(processes=min(8, args.trials)) as pool:
        results = pool.map(run_simulated_annealing_trial, params_list)

    print("All trials completed.")
