import json

class room:
    def __init__(self,x,y):
        self.current_people=x
        self.limit_people=y
        return None
    def leave(self,x):
        self.current_people=self.current_people-x
        return self.current_people
    def enter(self,x):
        self.current_people=self.current_people+x
        return self.current_people
    def check_exceed(self):
        return ((self.current_people)>(self.limit_people))
    def show_room(self):
        print("current:{} limit:{}\n".format(self.current_people,self.limit_people))
        return 0
    def people(self,mood=None):
        if mood=='c':
            return self.current_people
        elif mood=='l':
            return self.limit_people
        elif mood!=None:
            return None
        return self.current_people,self.limit_people


class Simulator:
    def __init__(self,path):
        with open(path,'r') as j_data:
            self.rooms_data=json.load(j_data)
        self.Rooms=[]
        for i in range(1,self.rooms_data["number"]+1):
            self.Rooms.append(room(self.rooms_data["room"+str(i)]["current_people"],self.rooms_data["room"+str(i)]["limit_people"]))
            #print(self.Rooms)
        return None
    def transfer(self,x_r,x_p,y_r):
        self.Rooms[x_r].leave(x_p)
        self.Rooms[y_r].enter(x_p)
        return 0
    def Show(self):
        print("population of room")
        for i in range(0,len(self.Rooms)):
            print("room {}".format(i+1))
            self.Rooms[i].show_room()
        return 0
    def people(self,x,mood=None):
        if mood=='e':
            return self.Rooms[x].check_exceed
        return self.Rooms[x].people(mood)


    

if __name__ == "__main__":
    Simu=Simulator('rooms.json')
    Simu.Show()
    Simu.transfer(0,20,1)
    Simu.Show()
    print(Simu.people(1,'e'))
