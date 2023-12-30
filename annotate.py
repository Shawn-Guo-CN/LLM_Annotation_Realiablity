import random
import json
import os
from typing import Dict, Any
from openai import OpenAI
from openai.types import CompletionChoice


MODEL = "gpt-3.5-turbo-1106"
PREAMBLE = """You are an expert summary rater. Given a TEXT (completed with a SUBREDDIT and a TITLE) and two summaries, SUMMARY 1 and SUMMARY 2, your role is to help the user to choose the better one between them."""


def construct_query_message(post: str, chosen: str, rejected: str):
    """Construct the query message for the GPT-3 API.

    Args:
        post: str, the post
        chosen: str, the chosen summary
        rejected: str, the rejected summary

    Returns:
        message: str, the query message
    """
    summary_list = [chosen, rejected]
    idx_list =  [0, 1]
    random.shuffle(idx_list)

    if idx_list[0] == 0:
        ans = 1
    else:
        ans = 2

    msg = f"""{post}\n\nSUMMARY 1: {summary_list[idx_list[0]]}\n\nSUMMARY 2: {summary_list[idx_list[1]]}\n\nPlease just strictly output a JSON string, which has following keys:\n\n- preference: int, 1 if you prefer SUMMARY 1, 2 if you prefer SUMMARY 2\n- reason: str, the brief (less than 50 words) reason why you give the above preference\n"""

    return msg, ans


def get_completions(message: str, api_key: str, n: int = 1):
    """Get the logprob of the message.

    Args:
      message: str, the message to be evaluated
      api_key: str, the API key
      n: int, the number of completions to generate

    Returns:
      logprob: float, the logprob of the message
    """
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PREAMBLE},
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        n=n,
    )
    return completion.choices


def annotate_tldr_post(post: Dict[str, str], api_key: str) -> int:
    """Annotate the post.

    Args:
        post: dict, the post dictionary of the following format
        {
            'post': a string of the post
            'chosen': a string of the chonse summary
            'rejected': a string of the rejected summary
        }
        api_key: str, the API key

    Returns:
        accuracy: int, 1 if the chosen summary is better than the rejected, 0
        if the rejected summary is better than the chosen, -1 if the annotator
        output is invalid
        query_msg: str, the query message
        result: str, the result from the annotator
    """

    post_text = post['post']
    chosen_text = post['chosen']
    rejected_text = post['rejected']

    query_msg, ans = construct_query_message(
        post_text, chosen_text, rejected_text
    )
    completions = get_completions(query_msg, api_key, n=1)
    result = completions[0].message.content

    try:
        result = json.loads(result)
        choice = int(result['preference'])
    except ValueError:
        return -1, query_msg, result

    if not choice in [1, 2]:
        return -1, query_msg, result
    elif choice == ans:
        return 1, query_msg, result
    else:
        return 0, query_msg, result
