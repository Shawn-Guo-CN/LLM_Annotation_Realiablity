from typing import Dict, Any
from datasets import load_dataset


def get_tldr_pair_dict(post: Dict[str, Any]):
    """Get the a dict which contains a pair of summaries for posts from TLDR.

    Args:
        post: a dictionary of the HuggingFace dataset format

    Returns:
      post_dict: a dictionary of the following format
      {
          'post': a string of the post
          'chosen': a string of the chonse summary
          'rejected': a string of the rejected summary
      }
    """
    post_text = f"""TTILE: {post['info']['title']}
    TEXT: {post['info']['post']}
    """
    chosen_text = post['summaries'][post['choice']]['text']
    chosen_text = chosen_text.replace('\n\n', ' ')
    rejected_text = post['summaries'][1 - post['choice']]['text']
    rejected_text = rejected_text.replace('\n\n', ' ')

    return {
      'post': post_text,
      'chosen': chosen_text,
      'rejected': rejected_text
    }


def get_tldr_post_list(split:str = 'validation'):
    """Get the list of posts from the TLDR dataset.

    The post in the return list is organised as follows:
    {
      'post': a string of the post
      'chosen': a string of the chonse summary
      'rejected': a string of the rejected summary
    }


    Args:
        split: str, one of ['train', 'validation', 'test']

    Returns:
        post_list: list of strings
    """
    dataset = load_dataset(
      'openai/summarize_from_feedback',
      'comparisons',
      split='validation'
    )

    post_list = []
    for post in dataset:
      post_list.append(get_tldr_pair_dict(post))

    return post_list
