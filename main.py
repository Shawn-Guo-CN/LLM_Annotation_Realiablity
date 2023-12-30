import argparse
import pickle
import time
from tqdm import tqdm

from utils import get_api_keys
from data import get_tldr_post_list
from annotate import annotate_tldr_post


def main(num_sample : int = 100, start_idx : int = 0):
    api_key_list = get_api_keys()
    assert len(api_key_list) > 40, 'Please provide at least 40 API key.'
    tldr_post_list = get_tldr_post_list()
    accuracy_list = []

    pbar = tqdm(total=num_sample)
    idx = start_idx
    full_acc_list = []
    fuLL_query_msg_list = []
    fuLL_result_list = []

    while len(accuracy_list) < num_sample:
        api_idx = idx % len(api_key_list)
        accuracy, query_msg, result = annotate_tldr_post(
            tldr_post_list[idx],
            api_key_list[api_idx],
        )

        full_acc_list.append(accuracy)
        fuLL_query_msg_list.append(query_msg)
        fuLL_result_list.append(result)

        if accuracy == 0 or accuracy == 1:
            accuracy_list.append(accuracy)
            pbar.update(1)
        idx += 1
        time.sleep(1.5)

    with open('logs/full_acc_list.pkl', 'wb') as f:
        pickle.dump(full_acc_list, f)
    with open('logs/fuLL_query_msg_list.pkl', 'wb') as f:
        pickle.dump(fuLL_query_msg_list, f)
    with open('logs/fuLL_result_list.pkl', 'wb') as f:
        pickle.dump(fuLL_result_list, f)

    print('accuracy:', sum(accuracy_list)/len(accuracy_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-sample', '-n', type=int, default=100,
        help='the number of samples to annotate'
    )
    parser.add_argument(
        '--start-idx', type=int, default=0,
        help='the index of the first sample to annotate'
    )
    args = parser.parse_args()

    main(
        num_sample=args.num_sample,
        start_idx=args.start_idx
    )
