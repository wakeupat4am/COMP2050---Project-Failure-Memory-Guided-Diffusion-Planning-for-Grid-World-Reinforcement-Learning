from experiments.run_comparison import run_comparison


def main():
    print("Running experiments...")
    df, csv_path = run_comparison()
    print(f"Saved efficiency comparison table to {csv_path}")
    print("\nFinal Efficiency Comparison:")
    print(df)


if __name__ == "__main__":
    main()
