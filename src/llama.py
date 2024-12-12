from enum import StrEnum
import asyncio
from inspect import Signature
import uvloop
from ollama import chat
from ollama import ChatResponse
import functools
from typing import Callable, Any, NamedTuple, TypedDict, Sequence, Literal
import dspy
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


def prompt() -> Message:
    prompt = """Find the examples in this page and return them in the following format: 
    { examples: [str, ...] }. 
    For example, if there was only one example like this: 
    For Example: 
        1,2,3, 
    the response would be: 
    { example: ["1,2,3"] }
    if on the other hand, there were two examples:
    For example:
        1,9,10
    ...
    for example:
        10, 10
        100, 1000
    the response would be:
    { example: ["1,9,10", "10,10\n100, 1000"]}
    """

    return Message(role="user", content=prompt)


def system_prompt() -> Message:
    return Message(
        role="system",
        content="You are a helpful AI that retrieves information from websites and returns answers as json.",
    )


def query(html: str) -> Message:
    return Message(
        role="user",
        content="""Here is the web-page, please find the example and return it as json. 
        It is very important that you get the answer exactly right.\n {0}""".format(
            html
        ),
    )


async def get_example(day: int) -> Sequence[str]:
    site = f"https://adventofcode.com/2024/day/{day}"
    html = await read_page(site)
    messages = [system_prompt(), prompt(), query(html)]
    resp = await get_response(messages, options())
    return resp.message.content["example"]


async def main():
    asyncio.ensure_future(get_example(2))


if __name__ == "__main__":
    uvloop.run(main())
