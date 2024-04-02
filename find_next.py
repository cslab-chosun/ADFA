

class findesec():
    
    def finde_second(list_i):
        mx = max(list_i[0], list_i[1])
        secondmax = min(list_i[0], list_i[1])
        secondmax_index=0
        mx_index=0
        n = len(list_i)
        for i in range(2,n):
            if list_i[i] > mx:
                secondmax = mx
                mx = list_i[i]
            elif list_i[i] > secondmax and \
                mx != list_i[i]:
                secondmax = list_i[i]
            elif mx == secondmax and \
                secondmax != list_i[i]:
                secondmax = list_i[i]
        list_return=[]
        for i in range(n):
            if list_i[i]==max(list_i):
                mx_index=i
                list_return.append(max(list_i))
                list_return.append(mx_index)
        for i in range(n):
            if list_i[i]==secondmax:
                secondmax_index=i
                list_return.append(secondmax)
                list_return.append(secondmax_index)
        return list_return

