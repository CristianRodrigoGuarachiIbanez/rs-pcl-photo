
def solution(S):
    # write your code in Python 3.6
    if len(S) == 0:
        return 0
    S = S.split(",")
    results = []
    for i in S:
        if i.isnumeric():
            if i.isdigit():
                my_num = int(i)
                results.append(my_num)
            else:
                pass
        else:
            pass
    return sum(results) // len(results)


if (__name__ == "__main__"):
    A = "3,4,5,6,99"
    smallest = solution(A)
    print(smallest)
