import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Test APUS Reflex with a local translator LLM.")
    parser.add_argument("--ask", metavar="QUESTION", help="answer one question in the terminal instead of launching the UI")
    args = parser.parse_args()

    if args.ask:
        from reflex_lab.reflex import Reflex
        from reflex_lab.translator import Translator

        request = Translator().translate(args.ask)
        print(json.dumps(request, indent=2, ensure_ascii=False))
        print(json.dumps(Reflex().decide(request), indent=2, ensure_ascii=False))
    else:
        from reflex_lab.app import launch

        launch()
