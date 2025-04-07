import random
import pandas as pd
import csv
import os
import timeit
from collections import defaultdict
import argparse


def calculate_travel_distance(schedule, distance_matrix):
    """Computes the total travel distance for a given schedule."""
    n_teams = len(schedule[0])
    rounds = len(schedule)

    dist_per_team = [0] * n_teams
    total_distance = 0

    for team_idx in range(n_teams):
        prev_opponent, prev_location = team_idx, 'home'

        for round in range(rounds):
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


def generate_random_row_first(n_teams):
    rounds = (n_teams - 1) * 2
    schedule_matrix = [[None] * n_teams for _ in range(rounds)]

    teams = list(range(-n_teams, n_teams + 1))
    teams.remove(0)

    for round in range(rounds):
        teams_to_pick = teams.copy()
        for team in range(n_teams):
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


def generate_random_column_first(n_teams):
    rounds = (n_teams - 1) * 2
    schedule_matrix = [[None] * n_teams for _ in range(rounds)]

    teams = list(range(-n_teams, n_teams + 1))
    teams.remove(0)

    for team in range(n_teams):
        team += 1
        teams_to_pick = teams.copy()
        teams_to_pick.remove(team)  # remove current team
        teams_to_pick.remove(-team)
        for round in range(rounds):  # pick opponent for each round
            opponent = random.choice(teams_to_pick)
            teams_to_pick.remove(opponent)

            schedule_matrix[round][team - 1] = opponent

    return schedule_matrix


def check_constraints_violated(schedule):
    n_teams = len(schedule[0])
    max_streak = 3

    violations = {"total": 0, "double_round_robin": 0, "noRepeat": 0, "maxStreak": 0}
    match_count = {}  # track how many times each matchup has occurred
    last_opponent = [-1] * n_teams  # store last opponent per team
    home_streaks = [0] * n_teams
    away_streaks = [0] * n_teams
    last_location = [None] * n_teams  # "home" or "away"

    required_matches = set()
    for team_idx in range(n_teams):
        for opponent_idx in range(team_idx + 1, n_teams):  # only generate unique pairs
            required_matches.add(frozenset([team_idx, opponent_idx]))  # Team X vs Team Y (unordered)

    # generate the home and away matches, need (team_idx, opponent_idx) and (opponent_idx, team_idx)
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
        for team_idx in range(n_teams):
            opponent = round_matches[team_idx]

            if opponent > 0:
                home, away = "opponent", "team"
            else:
                home, away = "team", "opponent"

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

def random_search(n_schedules):
    for schedule_type in ['row', 'column']:
        directory_path = "results/random_search/" + str(schedule_type) + '/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        overview_file = open(directory_path + "average.txt", "w")
        overview_writer = csv.writer(overview_file)
        overview_writer.writerow(
            ['n', 'schedule_type', 'distance', 'distance_per_team', 'constraints_violated', 'run_time'])

        best_file = open(directory_path + "best.txt", "w")
        best_writer = csv.writer(best_file)
        best_writer.writerow(
            ['n', 'schedule_type', 'distance', 'distance_per_team', 'constraints_violated', 'schedule'])

        worst_file = open(directory_path + "worst.txt", "w")
        worst_writer = csv.writer(worst_file)
        worst_writer.writerow(
            ['n', 'schedule_type', 'distance', 'distance_per_team', 'constraints_violated', 'schedule'])

        for n_teams in range(4, 51, 2):

            print(f"schedule type: {schedule_type}, n teams: {n_teams}")

            # distance matrix
            distance_matrix_path = "datasets/SM.csv"
            distance_matrix = pd.read_csv(distance_matrix_path, header=None).to_numpy(dtype=int)

            result_file = open(directory_path + str(n_teams) + ".txt", "w")
            result_writer = csv.writer(result_file)
            result_writer.writerow(
                ['n', 'schedule_type', 'distance', 'distance_per_team', 'constraints_violated', 'schedule'])

            best_schedule = [n_teams, schedule_type, 0, [], {}, []]
            worst_schedule = [n_teams, schedule_type, 0, [], {}, []]

            summed_distance = 0
            summed_dist_per_team = [0] * n_teams
            summed_constraints = defaultdict(int)

            start = timeit.default_timer()

            for j in range(n_schedules):
                # rand generate schedule
                if schedule_type == "row":
                    schedule = generate_random_row_first(n_teams)
                else:
                    schedule = generate_random_column_first(n_teams)

                # calc distance
                distance, distance_per_team = calculate_travel_distance(schedule, distance_matrix)
                summed_distance += distance
                summed_dist_per_team = [a + b for a, b in zip(summed_dist_per_team, distance_per_team)]

                # count constraints
                constraints_violated = check_constraints_violated(schedule)

                # sum all constrained violations
                for key, value in constraints_violated.items():
                    summed_constraints[key] += value

                # save instances in file
                result_writer.writerow(
                    [n_teams, schedule_type, distance, distance_per_team, constraints_violated, schedule])

                # save best, worst
                if distance < best_schedule[2] or j == 0:
                    best_schedule = [n_teams, schedule_type, distance, distance_per_team, constraints_violated,
                                     schedule]
                if distance > worst_schedule[2]:
                    worst_schedule = [n_teams, schedule_type, distance, distance_per_team, constraints_violated,
                                      schedule]

            avg_dist = summed_distance / n_schedules
            avg_dist_per_team = [x / n_schedules for x in summed_dist_per_team]
            avg_constraints_violated = {key: (summed_constraints[key] / n_schedules) if n_schedules > 0 else 0 for key
                                        in summed_constraints}

            stop = timeit.default_timer()
            time = stop - start
            overview_writer.writerow(
                [n_teams, schedule_type, avg_dist, avg_dist_per_team, avg_constraints_violated, time])

            best_writer.writerow(
                [n_teams, schedule_type, best_schedule[2], best_schedule[3], best_schedule[4], best_schedule[5]])

            worst_writer.writerow(
                [n_teams, schedule_type, worst_schedule[2], worst_schedule[3], worst_schedule[4], worst_schedule[5]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random schedule generator")
    parser.add_argument("--n_schedules", type=int, default=5000,
                        help="Number of random schedules to generate per team size (default: 5000)")
    args = parser.parse_args()

    random_search(args.n_schedules)
