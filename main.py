import time
import pickle
from tqdm import tqdm

from data import get_tldr_post_list
from annotate import annotate_tldr_post


def main():
    tldr_post_list = get_tldr_post_list()
    accuracy_list = []
    total_num = 500

    pbar = tqdm(total=total_num)
    idx = 0
    full_acc_list = []
    fuLL_query_msg_list = []
    fuLL_result_list = []
    while len(accuracy_list) < total_num:
        accuracy, query_msg, result = annotate_tldr_post(tldr_post_list[idx])
        full_acc_list.append(accuracy)
        fuLL_query_msg_list.append(query_msg)
        fuLL_result_list.append(result)

        if accuracy !=-1:
            accuracy_list.append(accuracy)
            pbar.update(1)
        idx += 1
        time.sleep(20)

    with open('logs/full_acc_list.pkl', 'wb') as f:
        pickle.dump(full_acc_list, f)
    with open('logs/fuLL_query_msg_list.pkl', 'wb') as f:
        pickle.dump(fuLL_query_msg_list, f)
    with open('logs/fuLL_result_list.pkl', 'wb') as f:
        pickle.dump(fuLL_result_list, f)

    print('accuracy:', sum(accuracy_list)/len(accuracy_list))


if __name__ == '__main__':
    main()
