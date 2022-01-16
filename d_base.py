import random

def load_database(file_name):
    end_list = []
    with open(file_name, "r") as file:
        for line in file:
            line = line[:-1]
            int_list = list(map(lambda x: int(x), line.split(":")))
            end_list.append(int_list)

    return end_list

def random_database_generator(num):
    file_name = "random_diet_database.txt"
    file = open(file_name, "w")
    meals_list = []
    lines = []
    for i in range(num):
        type_meal = str(random.randint(1, 3))
        meal = str(random.randint(1000, 9999))
        if meal in meals_list:
            continue
        meals_list.append(meal)
        prot = str(random.randint(10, 150))
        carbo = str(random.randint(20, 80))
        fats = str(random.randint(10, 50))
        line = type_meal + ":" + meal + ":" + prot + ":" + fats + ":" + carbo
        if i != num - 1:
            line = line + "\n"
        lines.append(line)
    file.writelines(lines)
    file.close()