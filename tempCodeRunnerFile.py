    for i in range(3):           
        plt.figure()
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
        
    plt.show()
    
    print(day_macros)
    except ValueError as ex:
        print(ex)
        continue