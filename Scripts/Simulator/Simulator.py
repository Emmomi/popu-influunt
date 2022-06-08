import json

class room:
    def __init__(self,x,y):
        self.current_people=x
        self.limit_people=y
    def leave(self,x):
        self.current_people=self.current_people-x
        return self.current_people
    def enter(self,x):
        self.current_people=self.current_people+x
        return self.current_people
    def check_exceed(self):
        return self.current_people>self.limit_people
    def show_room(self,a):
        if a==1:
            print("current:{} limit:{}\n".format(self.current_people,self.limit_people))
        return self.current_people,self.limit_people


class Simulator:
    def __init__(self):
        j_data=open('rooms.json','r')
        self.rooms_data=json.load(j_data)
        self.Rooms=[]
        for i in range(1,self.rooms_data["number"]):
            self.Rooms.append(room(self.rooms_data["room"+str(i)]["current_people"],self.rooms_data["room"+str(i)]["limit_people"]))
        return None
    def transfer(self,x_r,x_p,y_r):
        self.Rooms[x_r].leave(x_p)
        self.Rooms[y_r].enter(x_p)
        return 0
    def Show(self):
        for i in range(1,len(self.Rooms)):
            self.Rooms[i].show_room(1)
        return 0


    

if __name__ == "__main__":
    Simu=Simulator()
    #Simu.Show()
    Simu.transfer(0,20,1)
    Simu.Show()
