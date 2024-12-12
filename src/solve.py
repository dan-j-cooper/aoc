#!/usr/bin/env python

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal
import requests
from loguru import logger
import typer
import pathlib

import uvloop
from lib import *
from llama import get_example

app = typer.Typer(pretty_exceptions_show_locals=False)

session_token = "53616c7465645f5fbcdd459d720e0b7827797a8739272be070a3bcd8c1752371fbb7d6165aed156527b251e951152306ed6b88f2e4e249bc69c19de1a6a5bc59"
YEAR = 2024
INPUTS = pathlib.Path.cwd() / "puzzle-inputs"


def request_puzzle(year: int, day: int) -> str:
    url = f"https://adventofcode.com/{year}/day/{day}/input"
    jar = {"session": session_token}
    response = requests.get(url, cookies=jar)

    if response.status_code == 200:
        return response.text
    else:
        raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")


def read_till_quit():
    lines = []
    print("enter lines or type 'quit' to exit")
    while (line := input()) != "quit":
        lines.append(line)
    return "\n".join(lines)


def get_puzzle(
    day: int,
    part: Literal[1, 2],
    example: bool,
    year: int = YEAR,
):
    if not INPUTS.exists():
        INPUTS.mkdir()
    if example:
        parts = {i: INPUTS / f"day-{day}-part-{i}-example" for i in range(2)}
        if parts[part].exists()
            return parts[part].read_text()

        examples = asyncio.ensure_future(get_example(day)).result()
        for ex, f in zip(examples, parts.values()):
            f.write_text(ex)

        return examples[i]

    else:
        file = INPUTS / f"day-{day}"
        if file.exists():
            return file.read_text()
        puzzle = request_puzzle(year, day)
        _ = file.write_text(puzzle)

    return puzzle


_numbers = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "eighteen",
    19: "nineteen",
    20: "twenty",
}

for i in range(5):
    _numbers[20 + i + 1] = _numbers[20] + _numbers[i + 1]

solve = {
    d: {
        1: globals()[f"day_{_numbers[d]}_part_one"],
        2: globals()[f"day_{_numbers[d]}_part_two"],
    }
    for d in range(1, 2)
}

tpe = ThreadPoolExecutor(max_workers=8)

async def _main_loop():

@app.command()
def main(
    day: int,
    part: int,
    example: Annotated[bool, typer.Option("--example", "-e")] = False,
):
    uvloop.run(_main_loop())
    puzzle = get_puzzle(day, part, example)
    ans = solve[day][part](puzzle)
    print(f"the answer is: {ans}")


if __name__ == "__main__":
    app()
