import random
import copy
from numpy.lib.function_base import average
import d_base
from matplotlib import pyplot as plt
import numpy as np
import json
import itertools

def sched_diet_generator(meals_list, macro, wages, meals_schedule, days):
    menu = []
    existing_meals = []
    factors = []
    for i in range(days):
        for meal_wage in meals_schedule:
            error_factor = 100000
            curr_macro = [meal_wage * elem for elem in macro]
            for meal in meals_list:
                curr_factor = get_factor(curr_macro, meal[2:5], wages)
                #print(curr_factor)
                if curr_factor < error_factor and meal not in existing_meals:
                    error_factor = curr_factor
                    final_meal = meal
            factors.append(error_factor)
            menu.append(final_meal)
            existing_meals.append(final_meal)
        if i >= 10:
            del existing_meals[:len(meals_schedule)]
    print("Factors\n",factors)
    print("Menu\n",menu)
    return menu, factors

def sched_diet_generator_2(meals_list, macro, wages, meals_schedule, days):
    menu = []
    existing_meals = []
    factors = []
    begin_macro=copy.deepcopy(macro)
    macro_list=[]
    for i in range(days):
        end_macro=[0,0,0]
        # macro_list.append(macro)
        for meal_wage in meals_schedule:
            error_factor = 100000
            curr_macro = [meal_wage * elem for elem in macro]
            for meal in meals_list:
                curr_factor = get_factor(curr_macro, meal[2:5], wages)
                #print(curr_factor)
                if curr_factor < error_factor and meal not in existing_meals:
                    error_factor = curr_factor
                    final_meal = meal
            factors.append(error_factor)
            
            menu.append(final_meal)
            for j in range(3):
                end_macro[j]+=final_meal[2+j]
            existing_meals.append(final_meal)
        print("macro",macro,end_macro)
        macro_list.append(copy.deepcopy(macro))
        for j in range(3):
            tmp=macro[j]+macro[j]-end_macro[j]
            if tmp>= 1.2*begin_macro[j]:
                macro[j]=1.2*begin_macro[j]
            elif tmp<= 0.8*begin_macro[j]:
                macro[j]=0.8*begin_macro[j]
            else:
                macro[j]=tmp
        
        if i >= 10:
            del existing_meals[:len(meals_schedule)]
    print("Factors\n",factors)
    print("Menu\n",menu)
    print("\n\n\n",macro_list,"\n\n\n")
    return menu, factors,macro_list
def get_factor(ideal_macro, meal_macro, wages):
    diff_list = get_diff_for_macro(ideal_macro, meal_macro)
    return sum(list(map(lambda x, y: x * y, diff_list, wages)))

def get_diff_for_macro(ideal_macro, meal_macro):
    return [abs(ideal_macro_one - meal_macro_one) for ideal_macro_one, meal_macro_one in zip(ideal_macro, meal_macro)]


def random_diet_generator(meals_list, macro, wages, meals_schedule, days):
    menu = []
    existing_meals = []
    factors = []
    for i in range(days):
        for meal_wage in meals_schedule:
            curr_macro = [meal_wage * elem for elem in macro]
            while True:
                meal = random.choice(meals_list)
                curr_factor = get_factor(curr_macro, meal[2:5], wages)
                if meal not in existing_meals:
                    # print(curr_factor)
                    # print("-----")
                    # print(meal)
                    # print(curr_macro)
                    factors.append(curr_factor)
                    menu.append(meal)
                    existing_meals.append(meal)
                    print(curr_macro)
                    break
        if i >= 10:
            del existing_meals[:len(meals_schedule)]
    print(" Factors\n",factors)
    print("Menu\n",menu)
    return menu,factors



def greedy_diet_generator(meals_list, macro, wages, meals_schedule, days):
    dic_macro = {0: [], 1: [], 2: []}
    dic_factors = {0: [], 1: [], 2: []}
    for c in range(3):
        menu = []
        existing_meals = []
        factors = []
        criterion = c # 0 - prot, 1 - fats, 2 - carbo
        criterion_list = []
        for i in range(days):
            for meal_wage in meals_schedule:
                curr_macro = [meal_wage * elem for elem in macro]
                for meal in meals_list:
                    diff_macro = get_diff_for_macro(curr_macro, meal[2:5])
                    criterion_list.append(diff_macro[criterion])
                while True:
                    if meals_list[criterion_list.index(min(criterion_list))] in existing_meals:
                        criterion_list[criterion_list.index(min(criterion_list))] = 1000
                    else:
                        final_meal = meals_list[criterion_list.index(min(criterion_list))]
                        curr_factor = get_factor(curr_macro, final_meal[2:5], wages)
                        break
                factors.append(curr_factor)
                menu.append(final_meal)
                existing_meals.append(final_meal)
                criterion_list = []
            if i >= 10:
                del existing_meals[:len(meals_schedule)]
        dic_macro[criterion] = menu
        dic_factors[criterion] = factors
    print("Factors\n",dic_factors[1])
    print("Menu\n",dic_macro[1])
    return dic_macro[1],dic_factors[1]


def algo_type_choice(meals_list, macro, wages, meals_schedule):
    days = 5
    #ch = 2
    for ch in range(1, 5):
        if ch == 1:
            d1=random_diet_generator(meals_list, macro, wages, meals_schedule, days)[0]
            print("Koniec")
        if ch == 2:
            d2=greedy_diet_generator(meals_list, macro, wages, meals_schedule, days)[0]
            print("Koniec")
        if ch == 3:
            d3=sched_diet_generator(meals_list,macro, wages, meals_schedule, days)[0]
            print("Koniec")
        if ch == 4:
            d4=sched_diet_generator_2(meals_list,macro, wages, meals_schedule, days)[0]
    return [d1,d2,d3,d4]
def algo_type_choice_factors(meals_list, macro, wages, meals_schedule):
    days = 5
    #ch = 2
    for ch in range(1, 5):
        if ch == 1:
            d1=random_diet_generator(meals_list, macro, wages, meals_schedule, days)[1]
            
        if ch == 2:
            d2=greedy_diet_generator(meals_list, macro, wages, meals_schedule, days)[1]
            
        if ch == 3:
            d3=sched_diet_generator(meals_list,macro, wages, meals_schedule, days)[1]
        if ch == 4:
            d4=sched_diet_generator_2(meals_list,macro, wages, meals_schedule, days)[1]
        
            
    return [d1,d2,d3,d4]
def res_macros_type_choice(meals_list, macro, wages, meals_schedule):
    days = 5
    return sched_diet_generator_2(meals_list,macro, wages, meals_schedule, days)[2]
        
            
    return [d1,d2,d3,d4]
def diet(meals_list, prot, fats, carbo, meals_num): #type od diet 1 or 2 or 3
    macro = [prot, fats, carbo]
    print(macro)
    wages = [0.4, 0.25, 0.35] #[proteins, fats, carbo]
    three_meals_schedule = [0.25, 0.55, 0.20]
    four_meals_schedule = [0.2, 0.5, 0.1, 0.2]
    five_meals_schedule = [0.15, 0.1, 0.4, 0.15, 0.2]
    if meals_num == 3:
        meals_schedule = three_meals_schedule
    elif meals_num == 4:
        meals_schedule = four_meals_schedule
    else:
        meals_schedule = five_meals_schedule
    return algo_type_choice(meals_list, macro, wages, meals_schedule)
def factors(meals_list, prot, fats, carbo, meals_num): #type od diet 1 or 2 or 3
    macro = [prot, fats, carbo]
    print(macro)
    wages = [0.4, 0.25, 0.35] #[proteins, fats, carbo]
    three_meals_schedule = [0.25, 0.55, 0.20]
    four_meals_schedule = [0.2, 0.5, 0.1, 0.2]
    five_meals_schedule = [0.15, 0.1, 0.4, 0.15, 0.2]
    if meals_num == 3:
        meals_schedule = three_meals_schedule
    elif meals_num == 4:
        meals_schedule = four_meals_schedule
    else:
        meals_schedule = five_meals_schedule
    return algo_type_choice_factors(meals_list, macro, wages, meals_schedule)
def res_diet_macros(meals_list, prot, fats, carbo, meals_num): #type od diet 1 or 2 or 3
    macro = [prot, fats, carbo]
    print(macro)
    wages = [0.4, 0.25, 0.35] #[proteins, fats, carbo]
    three_meals_schedule = [0.25, 0.55, 0.20]
    four_meals_schedule = [0.2, 0.5, 0.1, 0.2]
    five_meals_schedule = [0.15, 0.1, 0.4, 0.15, 0.2]
    if meals_num == 3:
        meals_schedule = three_meals_schedule
    elif meals_num == 4:
        meals_schedule = four_meals_schedule
    else:
        meals_schedule = five_meals_schedule
    return res_macros_type_choice(meals_list, macro, wages, meals_schedule)

def type_of_meals(list):
    omni_diet_list = []
    veget_diet_list = []
    vegan_diet_list = []
    for i in range(len(list)):
        if list[i][0] == 1:
            omni_diet_list.append(list[i])
        elif list[i][0] == 2:
            veget_diet_list.append(list[i])
        else:
            vegan_diet_list.append(list[i])

    return omni_diet_list, veget_diet_list, vegan_diet_list

if __name__ == '__main__':
    while True:
    # try:
        # print("Hello")
        # print("Pass your micronutrients:")
        file = "random_diet_database.txt"
        protein = 200 #int(input("Protein: "))
        carbohydrates = 300 #int(input("Carbohydrates: "))
        fats = 100 #int(input("Fats: "))
        meals_num = 5 #int(input("How many meals do you want? "))
        type_of_meal = 3 #int(input("Which type of diet you prefer (pass the number):\n[1] Omnivorous diet \n[2] Vegetarian diet\n[3] Vegan diet\n5"))
        #d_base.random_database_generator(1500)
        data = d_base.load_database(file)
        meat_diet_list, vegetarian_diet_list, vegan_diet_list = type_of_meals(data)
        omni_diet_list = meat_diet_list + vegetarian_diet_list + vegan_diet_list
        vegetarian_diet_list.extend(vegan_diet_list)
        if type_of_meal == 1:
            variants=diet(omni_diet_list, protein, fats, carbohydrates, meals_num)
            l_factors=factors(omni_diet_list, protein, fats, carbohydrates, meals_num)
            reserve_macros=res_diet_macros(omni_diet_list, protein, fats, carbohydrates, meals_num)
        elif type_of_meal == 2:
            variants=diet(vegetarian_diet_list, protein, fats, carbohydrates, meals_num)
            l_factors=factors(vegetarian_diet_list, protein, fats, carbohydrates, meals_num)
            reserve_macros=res_diet_macros(vegetarian_diet_list, protein, fats, carbohydrates, meals_num)
        else:
            variants=diet(vegan_diet_list, protein, fats, carbohydrates, meals_num)
            l_factors=factors(vegan_diet_list, protein, fats, carbohydrates, meals_num)
            reserve_macros=res_diet_macros(vegan_diet_list, protein, fats, carbohydrates, meals_num)
        break
    variants_ret=copy.deepcopy(variants)
    day_macros=[]
    days_num=5
    lc=0
    for x in variants:
        
        day_macros.append([])
        for i in range(days_num):
            day_macros[lc].append([0,0,0])
            for j in range(i*meals_num,(i+1)*meals_num):
                day_macros[lc][i][0]+=x[j][2]
                day_macros[lc][i][1]+=x[j][3]
                day_macros[lc][i][2]+=x[j][4]
        lc+=1
    print()
    
    lc=0
    xpoints=np.array([x for x in range(days_num)])
    y_ideal=np.array([protein for x in range(days_num)])
    ideal_proteins=np.array([protein for x in range(days_num)])
    ideal_fats=np.array([fats for x in range(days_num)])
    ideal_carbs=np.array([carbohydrates for x in range(days_num)])
    ideal=[ideal_proteins,ideal_fats,ideal_carbs]
    labels=["random","greedy","factor"]
    titles=["Protein", "Fats", "Carbs"]    
    # for x in day_macros:
    #     print(x)
    # for x in l_factors:
    #     print(x)
    print()
    avg_factors=[]
    for x in l_factors:
        tmp_avg_factors=[]
        for i in range(days_num):
            tmp_avg_factors.append(average(x[i*days_num:(i+1)*days_num]))
        avg_factors.append(tmp_avg_factors)
    reserve_variant=variants.pop(3)
    reserve_variant_factors=avg_factors.pop(3)
    reserve_day_macros=day_macros.pop(3)
    # PLOTS
    plt.subplot(2,2,1)
    plt.title("Factors")
    factor_xpoints=np.array([x for x in range(days_num)])
    plt.plot(factor_xpoints,np.array([0 for x in range(days_num)]),'o', label="ideal factor")
    for x in avg_factors:
        tmp_ypoints=np.array(x)
        plt.plot(factor_xpoints,tmp_ypoints,'o',label=labels[lc])
        lc+=1
        tmp_ypoints=[]
    lc=0
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.show()
    for i in range(3): 
        
        # plt.figure()
        plt.subplot(2,2,i+2)
        plt.title(titles[i])
        plt.plot(xpoints,ideal[i],'o',label="ideal macro")
        for x in day_macros:
            ypoints_tmp=[]          
            for y in x:
                ypoints_tmp.append(y[i])
            ypoints_tmp=np.array(ypoints_tmp)
            plt.plot(xpoints,ypoints_tmp,'o',label=labels[lc%3])
            lc+=1
            ypoints_tmp=[]
            
            
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(xpoints,np.array([0 for x in range(days_num)]),'o', label="ideal factor")
    lc=0
    # for x in reserve_variant_factors:
    #     tmp_ypoints=np.array(x)
    #     plt.plot(xpoints,tmp_ypoints,'o',label=labels[lc])
    #     lc+=1
    #     tmp_ypoints=[]
    plt.title("Factors")
    plt.plot(xpoints,np.array(reserve_variant_factors),'o',label="reserve factor")
    res_ideal=[[],[],[]]
    np_res_ideal=[]

    for y in reserve_macros:
        for i in range(3):
            res_ideal[i].append(y[i])
        # res_ideal[i]=np.array(res_ideal[i])
    # for x in res_ideal:
    #     np_res_ideal.append(np.array(x))
    # np_res_ideal.append(np.array([np.concatenate(i) for i in res_ideal[0]]))
    res_proteins=np.array(res_ideal.pop(0))
    res_fats=np.array(res_ideal.pop(0))
    res_carbs=np.array(res_ideal.pop(0))
    np_res_ideal=[res_proteins,res_fats,res_carbs]
    
    print("b ",res_carbs, "e")
    print(ideal, np_res_ideal)
    # print("elo",np_res_ideal)
    # print(reserve_macros)

    for i in range(3):  
        # plt.figure()
        plt.subplot(2,2,i+2)
        plt.title(titles[i])
        plt.plot(xpoints,np_res_ideal[i],'o',label="ideal macro") 
        ypoints_tmp=[]     
        for x in reserve_day_macros:
            ypoints_tmp.append(x[i])
        ypoints_tmp=np.array(ypoints_tmp)
        plt.plot(xpoints,ypoints_tmp,'o',label="reserve factor")   
        ypoints_tmp=[]        
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()    
        
    # plt.show()
    json_data=[]
    variants_ret_2=[]
    tmp=0
    for x in variants_ret:
        variants_ret_2.append([])
        
        for i in range(0,meals_num*days_num,days_num):
            variants_ret_2[tmp].append(x[i:i+meals_num])
        tmp+=1
    print(variants_ret_2)
    #JSON convert
    tmp2=0
    tmp3=0
    tmp4=0
    names_macros=itertools.cycle(["proteins","fats","carbs"])
    labels_2=["random","greedy","factor","reserve factor"]
    days_it=itertools.cycle(["Day"+str(x) for x in range(1,days_num+1)])
    labels_it = itertools.cycle(labels_2)
    for x in variants_ret_2:
        json_data.append({"name": next(labels_it), "children": []})
        for y in x:
            json_data[tmp2]["children"].append({"name": next(days_it), "children":[]})
            for z in y:
                json_data[tmp2]["children"][tmp3]["children"].append({"name": z[1], "size": z[2]+z[3]+z[4] , "children":[]})
                for zz in range(3):
                    json_data[tmp2]["children"][tmp3]["children"][tmp4]["children"].append({"name": next(names_macros)+"\n"+str(z[zz+2]), "size": z[zz+2]})
                tmp4+=1
            tmp4=0
                
            tmp3+=1
        tmp3=0
        tmp2+=1
    # json_data=dict(json_data)
    json_data_2={"name":"Diet Plan", "children":json_data}
    # print(json_data_2)
    with open ('test.json','w') as f:
        json.dump(json_data_2,f,indent=4)



            