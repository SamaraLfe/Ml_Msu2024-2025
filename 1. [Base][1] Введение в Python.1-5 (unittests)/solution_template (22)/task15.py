from typing import List

def hello(x = None) -> str:
    if x == None or x == "": return "Hello!"
    else: return  "Hello, "+str(x)+"!"

def int_to_roman(x: int) -> str:
    list = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    roman_out = ""
    for (num, roman) in list:
        while x >= num:
            roman_out += roman
            x -= num
    return  roman_out

def longest_common_prefix(strs_input: List[str]) -> str:
    strs_input = [x.lstrip() for x in strs_input]
    if not strs_input: return ""
    shortest_str = min(strs_input, key=len)
    for i, subchar in enumerate(shortest_str):
        for s in strs_input:
            if s[i] != subchar: return shortest_str[:i]
    return shortest_str

def primes() -> int:
    i = 2
    while True:
        flag = True
        for j in range(2,int(i**0.5)+1):
            if i % j == 0:
                flag = False
                break
        if flag:
            yield i
        i+= 1

class BankCard:
    def __init__(self, total_sum, balance_limit=None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def __call__(self, sum_spent):
        if sum_spent > self.total_sum:
            raise ValueError(f"Not enough money to spend {sum_spent} dollars.")
        else:
            self.total_sum -= sum_spent
            print(f"You spent {sum_spent} dollars.")

    def __repr__(self):
        return "To learn the balance call balance."

    def __add__(self, other):
        total_sum = self.total_sum + other.total_sum
        balance_limit = max(self.balance_limit, other.balance_limit)
        return BankCard(total_sum, balance_limit)

    @property
    def balance(self):
        if self.balance_limit is not None:
            if self.balance_limit == 0:
                raise ValueError("Balance check limits exceeded.")
            else:
                self.balance_limit -= 1
        return self.total_sum

    def put(self, sum_put):
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars.")

