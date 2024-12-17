from enum import StrEnum
import asyncio
from inspect import Signature
import uvloop
from ollama import chat
from ollama import ChatResponse
import functools
import orjson
from typing import Callable, Any, NamedTuple, TypedDict, Sequence, Literal
import bs4
import requests

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]


async def run_in_loop[T](blocking_fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(blocking_fn, *args, **kwargs),
    )


async def read_page(url: str) -> str:
    r = requests.get(url)
    if r.status_code == 200:
        html = r.content
        soup = bs4.BeautifulSoup(html, "html.parser")
        return soup.prettify()
    else:
        raise RuntimeError(f"unable to read from page got {r.status_code}")


async def get_response(messages: Dialog, options: dict[str, Any]) -> ChatResponse:
    response = await run_in_loop(
        chat,
        model="llama3.3:70b",
        messages=messages,
        options=options,
        format="json",
    )
    return response


def options() -> dict[str, Any]:
    return {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 100,
        "stop": ["<|end_of_text|>"],
        "num_ctx": int(32e3),
    }


def system_prompt() -> Message:
    return Message(
        role="system",
        content="You are a helpful AI that retrieves information from websites and returns answers to queries in json.",
    )


def prompt() -> Message:
    prompt = """Find the Nth example in this page and return it in the following format: 
    { examples: str }. 
    For example, if there was only one example in the document and the query was find part 1:
    For Example: 
        1,2,3, 
    the response would be: 
    { example: ["1,2,3"] }
    if on the other hand, there were two examples in the html:
    For example:
        1,9,10
    ...
    for example:
        10, 10
        100, 1000
    and the query was: find example 1, response would be:
    { example: "1,9,10" },
    if, on the other hand, the query had been: find example 2, 
    the response would be:
    { example: "10,10\n100, 1000"}
    It is very important that you accurately transcribe the exact example requested only.
    If no such example exists, or you fail to find the requested example, or you can't determine what should go into the example, you must put the
    word: Error in the response, ie:
    { example: "Error" }
    """

    return Message(role="user", content=prompt)


def query(html: str, part: Literal[1, 2]) -> Message:
    return Message(role="user", content=f"""find example{part}\n{html}""")


async def get_example(day: int, part: int) -> str:
    site = f"https://adventofcode.com/2024/day/{day}"
    html = await read_page(site)
    messages = [system_prompt(), prompt(), query(html, part)]
    resp = await get_response(messages, options())
    return orjson.loads(resp.message.content)["example"]


async def main():
    asyncio.ensure_future(get_example(2))


if __name__ == "__main__":
    uvloop.run(main())
