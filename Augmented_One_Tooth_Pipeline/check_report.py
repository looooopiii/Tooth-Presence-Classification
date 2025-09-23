import pandas as pd

rep = pd.read_csv("/home/user/lzhou/week8/runs/all_cpu_baseline/test_report_per_tooth.csv")

print("\n=== minimum supporting for 6 teeth ===")
print(rep.sort_values("support").head(6)[["tooth","support","F1","best_th"]])

print("\n=== minimum F1 for 6 teeth ===")
print(rep.sort_values("F1").head(6)[["tooth","support","F1","best_th"]])

print("\n=== maximum F1 for 6 teeth ===")
print(rep.sort_values("F1", ascending=False).head(6)[["tooth","support","F1","best_th"]])