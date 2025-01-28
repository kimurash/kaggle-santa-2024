import os

import pandas as pd
import slackweb
from dotenv import load_dotenv
from metrics.calc import score as calc_score
from optimizer import Optimizer


def notify_slack(message: str):
    load_dotenv()

    WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    slack = slackweb.Slack(url=WEBHOOK_URL)
    slack.notify(text=message)


if __name__ == "__main__":
    solution = pd.read_csv("../input/sample_submission.csv")

    submission = solution.copy(deep=True)
    optimizer = Optimizer()

    for idx, row in solution.iterrows():
        text = row["text"]

        opt_text = optimizer.optimize(text)
        submission.loc[idx, "text"] = opt_text

        notify_slack(f"text {idx}")

    score = calc_score(
        solution=solution,
        submission=submission,
        row_id_column_name="id",
    )

    with open("../output/score.txt", "w") as f:
        f.write(f"{score:.5f}")

    submission.to_csv("../output/submission.csv", index=False)

    notify_slack("santa-2024")
