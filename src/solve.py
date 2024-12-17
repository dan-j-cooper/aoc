#!/usr/bin/env python

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal
import aiohttp
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


def file_cache(path: pathlib.Path, cache={}) -> dict[str, pathlib.Path]:
    if not path.exists():
        path.mkdir()
    if not cache:
        cache = {f.name: f for f in path.iterdir()}
    return cache


async def request_puzzle(day: int, part: Literal[1, 2]) -> str:
    url = f"https://adventofcode.com/{YEAR}/day/{day}/input"
    jar = {"session": session_token}
    async with aiohttp.ClientSession() as sesh:
        async with sesh.get(url, cookies=jar) as reponse:
            if response.status_code == 200:
                return response.text
            else:
                raise RuntimeError(
                    f"Request failed: {response.status_code}, {response.text}"
                )


async def _get_example(day: int, part: Literal[1, 2]) -> str:
    file_name = f"day-{day}-part-{part}-example"
    cache = file_cache(INPUTS)

    if file_name in cache:
        example = cache[file_name].read_text()
    else:
        example = await get_example(day, part)
        f = INPUTS / file_name
        f.write_text(example)

    return example


async def _get_day(day: int, part: Literal[1, 2]) -> str:
    cache = file_cache(INPUTS)
    file_name = f"day-{day}"
    if file_name in cache:
        puzzle = cache[file_name].read_text()
    else:
        puzzle = await request_puzzle(day, part)
        (INPUTS / file_name).write_text(puzzle)
    return puzzle


async def get_puzzle(
    day: int,
    part: Literal[1, 2],
    example: bool,
):
    if example:
        puzzle = await _get_example(day, part)
    else:
        puzzle = await _get_day(day, part)

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


@app.command()
def main(
    day: int,
    part: int,
    example: Annotated[bool, typer.Option("--example", "-e")] = False,
):
    uvloop.run(_main(day, part, example))


async def _main(day: int, part: int, example: bool):
    puzzle = await get_puzzle(day, part, example)
    ans = solve[day][part](puzzle)
    print(f"the answer is: {ans}")


if __name__ == "__main__":
    app()
