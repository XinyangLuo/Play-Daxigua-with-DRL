class Fruit():
    def __init__(self):
        pass

class Grape(Fruit):
    def __init__(self):
        self.type = 1
        self.r = 7
        self.score = 0
        self.color = '#7e1671'

class Cherry(Fruit):
    def __init__(self):
        self.type = 1
        self.r = 10
        self.score = 1
        self.color = '#2e317c'

class Orange(Fruit):
    def __init__(self):
        self.type = 1
        self.r = 14
        self.score = 2
        self.color = '#207f4c'

class Lemon(Fruit):
    def __init__(self):
        self.type = 1
        self.r = 16
        self.score = 3
        self.color = '#f1ca17'

class Kiwi(Fruit):
    def __init__(self):
        self.type = 1
        self.r = 20
        self.score = 4
        self.color = '#fc8c23'

class Tomato(Fruit):
    def __init__(self):
        self.type = 1
        self.r = 24
        self.score = 5
        self.color = '#f13c22'

class Peach(Fruit):
    def __init__(self):
        self.type = 1
        self.r = 25
        self.score = 6
        self.color = '#5a191b'

class PineApple(Fruit):
    def __init__(self):
        self.type = 1
        self.r = 34
        self.score = 7
        self.color = '#8adb3b'

class Coco(Fruit):
    def __init__(self):
        self.type = 1
        self.r = 40
        self.score = 8
        self.color = '#807a7a'

class WaterMelon(Fruit):
    def __init__(self):
        self.type = 1
        self.r = 56
        self.score = 9
        self.color = '#ff83ac'

class BigWaterMelon(Fruit):
    def __init__(self):
        self.type = 1
        self.r = 56
        self.score = 110
        self.color = '#000000'

def create_fruit(type):
    if type == 1:
        return Grape()
    elif type == 2:
        return Cherry()
    elif type == 3:
        return Orange()
    elif type == 4:
        return Lemon()
    elif type == 5:
        return Kiwi()
    elif type == 6:
        return Tomato()
    elif type == 7:
        return Peach()
    elif type == 8:
        return PineApple()
    elif type == 9:
        return Coco()
    elif type == 10:
        return WaterMelon()
    elif type == 11:
        return BigWaterMelon()