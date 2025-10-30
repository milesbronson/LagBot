#!/usr/bin/env python3
"""Dashboard generator"""
import argparse
from src.training.dashboard import TrainingDashboard

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, help='Generate for specific run')
parser.add_argument('--compare', action='store_true', help='Generate comparison')
parser.add_argument('--report', action='store_true', help='Generate HTML report')
parser.add_argument('--all', action='store_true', help='Generate all')
args = parser.parse_args()

dashboard = TrainingDashboard()
runs = dashboard.dashboard.load_all_runs()

if not runs:
    print("No runs found")
    exit()

if args.run:
    dashboard.plot_single_run(args.run, f"dashboard_{args.run}.png")
elif args.compare:
    if len(runs) > 1:
        dashboard.plot_comparison("dashboard_comparison.png")
elif args.report:
    dashboard.generate_html_report()
elif args.all:
    for r in runs:
        dashboard.plot_single_run(r, f"dashboard_{r}.png")
    if len(runs) > 1:
        dashboard.plot_comparison("dashboard_comparison.png")
    dashboard.generate_html_report()
else:
    print("Use: --run NAME, --compare, --report, or --all")