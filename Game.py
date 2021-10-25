import Fruit
import pymunk.pygame_util
import pymunk
import numpy as np
from pygame.color import *
from pygame.locals import *
from pygame.key import *
import pygame
import random

from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


class Game():
    def __init__(self, FPS=60, stepTime=0.1, Exact=20, auxiliaryReward=False):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 900.0)

        pygame.init()
        self.WIDTH = 188
        self.HEIGHT = 350
        self.BOUND = 256
        self.roundTime = 0
        self.NEXT = 1
        self.FPS = FPS
        self.COLLISIONGAP = 0
        self.LOCK = False
        self.score = 0
        self.prevScore = 0
        self.length = 0
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.auxiliaryReward = auxiliaryReward
        self.states = []
        self.actions = []
        self.rewards = []

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.balls = []
        if stepTime == 'FPS':
            self.stepTime = 1/FPS
            #self.FPS = 60
        else:
            self.stepTime = stepTime
            #self.FPS = 1/stepTime
        self.Exact = Exact

        static_lines = [
            pymunk.Segment(self.space.static_body,
                           (0, 0), (0, self.HEIGHT), 0.0),
            pymunk.Segment(self.space.static_body, (0, self.HEIGHT),
                           (self.WIDTH, self.HEIGHT), 0.0),
            pymunk.Segment(self.space.static_body, (self.WIDTH,
                                                    self.HEIGHT), (self.WIDTH, 0), 0.0)
        ]

        for line in static_lines:
            line.elasticity = 0.1
            line.friction = 1
            self.space.add(line)

    def create_ball(self, i, pos):
        fruit = Fruit.create_fruit(i)
        r = fruit.r
        mass = np.pi*r**2
        inertia = pymunk.moment_for_circle(mass, 0, r, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = pos
        shape = pymunk.Circle(body, r, (0, 0))
        shape.elasticity = 0.1
        shape.friction = 1
        shape.collision_type = i
        self.space.add(body, shape)
        self.balls.append(shape)

    def my_events(self, click=False):
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONDOWN and click:
                self.roundTime = 0
                loc = pygame.mouse.get_pos()
                r = self.score - self.prevScore
                s = self.getImageState()
                a = loc[0]
                self.create_ball(self.NEXT, (loc[0], self.HEIGHT - self.BOUND))
                self.NEXT = random.randint(1, 5)
                self.prevScore = self.score
                self.states.append(s)
                self.actions.append(a)
                self.rewards.append(r)
                #print('action = {}, reward = {}'.format(a, r))
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    state = self.getState()
                    np.save('./saves/save', state)
            if event.type == QUIT:
                exit()

    def take_action(self, action=None):
        if not action is None:
            self.roundTime = 0
            r = Fruit.create_fruit(self.NEXT).r
            if action < r:
                action = r
            elif action > self.WIDTH - r:
                action = self.WIDTH-r
            self.create_ball(self.NEXT, (action, self.HEIGHT - self.BOUND))
            self.NEXT = random.randint(1, 5)
            self.length += 1

    def clear_screen(self):
        self.screen.fill(THECOLORS["white"])

    def draw_objects(self):
        self.space.debug_draw(self.draw_options)

    def setup_collision_handler(self):
        def collision(arbiter, space, data):
            if self.COLLISIONGAP == 0:
                x1, y1 = arbiter.shapes[0].body.position
                x2, y2 = arbiter.shapes[1].body.position
                i = arbiter.shapes[0].collision_type + 1
                if y1 > y2:
                    x, y = x1, y1
                else:
                    x, y = x2, y2
                if arbiter.shapes[0] in self.balls:
                    self.space.remove(
                        arbiter.shapes[0], arbiter.shapes[0].body)
                    self.balls.remove(arbiter.shapes[0])
                if arbiter.shapes[1] in self.balls:
                    self.space.remove(
                        arbiter.shapes[1], arbiter.shapes[1].body)
                    self.balls.remove(arbiter.shapes[1])

                self.create_ball(i, (x, y))
                self.COLLISIONGAP = 0.05
                self.score += Fruit.create_fruit(i).score

        for i in range(1, 11):
            self.space.add_collision_handler(i, i).post_solve = collision

    @staticmethod
    def get_distance(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def auxiliary_reward(self):
        lastBall = self.balls[-1]
        lastR = lastBall.radius
        lastPos = lastBall.body.position
        lasttype = lastBall.collision_type
        auxiliaryReward = 0
        for ball in self.balls[:-1]:
            r = ball.radius
            pos = ball.body.position
            t = ball.collision_type
            d = self.get_distance(pos, lastPos)
            if d <= r + lastR + 1 and lasttype + 1 == t:
                auxiliaryReward += 0.5
        return auxiliaryReward*self.auxiliaryReward

    def update(self):
        for ball in self.balls:
            pos = ball.body.position
            i = ball.collision_type
            r = ball.radius
            color = Fruit.create_fruit(i).color

            pygame.draw.circle(self.screen, color, pos, r)
        pygame.draw.line(self.screen, '#FF0000', (0, self.HEIGHT -
                                                  self.BOUND), (self.WIDTH, self.HEIGHT-self.BOUND))
        nextR = Fruit.create_fruit(self.NEXT).r
        nextColor = Fruit.create_fruit(self.NEXT).color
        pygame.draw.circle(self.screen, nextColor, (nextR, nextR), nextR)

    def check(self):
        if self.roundTime > 3:
            for ball in self.balls:
                if ball.body.position.y - ball.radius < self.HEIGHT-self.BOUND:
                    self.LOCK = True

    def runByClick(self, save=True):
        self.setup_collision_handler()
        while True:
            for i in range(self.Exact):
                step = self.stepTime/self.Exact
                self.space.step(step)
                self.roundTime += step
                self.COLLISIONGAP = max(self.COLLISIONGAP-step, 0)

            self.check()
            self.my_events(click=True)
            self.clear_screen()
            self.draw_objects()
            self.update()
            pygame.display.flip()
            self.clock.tick(self.FPS)
            pygame.display.set_caption(
                "Score = {}, Previous Score = {}".format(self.score, self.prevScore))
            if self.LOCK:
                print('Game Over, Score:{}'.format(self.score))
                self.states.append(self.getImageState())
                self.rewards.append(self.score - self.prevScore)
                if save:
                    self.saveTrajectory(25)
                    print('Trajectory saved')
                break

    def saveTrajectory(self, i=1):
        s = np.array(self.states)
        a = np.array(self.actions)
        r = np.array(self.rewards[1:])
        np.save('./trajectories/states{}'.format(i), s)
        np.save('./trajectories/actions{}'.format(i), a)
        np.save('./trajectories/rewards{}'.format(i), r)



    def NextState(self, action=None):
        self.take_action(action)
        while True:
            for x in range(self.Exact):
                step = self.stepTime/self.Exact
                self.space.step(step)
                self.roundTime += step
                self.COLLISIONGAP = max(self.COLLISIONGAP-step, 0)

            self.check()
            self.my_events()
            self.clear_screen()
            self.draw_objects()
            self.update()
            pygame.display.flip()
            self.clock.tick(self.FPS)

            if self.roundTime >= 3:
                reward = self.score - self.prevScore
                reward += self.auxiliary_reward()
                self.prevScore = self.score
                break

        return reward

    def getState(self):
        state = np.zeros((13, 101))
        state[self.NEXT-1, 0] = 1
        idx = 0
        for ball in self.balls:
            x, y = ball.body.position
            t = ball.collision_type
            state[t-1, idx%100+1] = 1
            state[-1, idx%100+1] = y
            state[-2, idx%100+1] = x
            idx += 1
        return state


    def getImageState(self):
        image = pygame.surfarray.pixels3d(pygame.display.get_surface())
        return image

    def loadGame(self):
        state = np.load('./saves/save.npy')
        for i in range(11):
            if state[i, 0] == 1:
                self.NEXT = i
                break
        for j in range(1, state.shape[1]):
            col = state[:, j]
            if np.all(col == 0):
                break
            for i in range(11):
                if col[i] == 1:
                    t = i + 1
                    break
            pos = (col[-2], col[-1])
            self.create_ball(t, pos)

    def reset(self):
        for ball in self.balls:
            self.space.remove(ball, ball.body)
        del self.balls
        self.balls = []
        self.score = 0
        self.prevScore = 0
        self.LOCK = False
        self.NEXT = 1
        self.COLLISIONGAP = 0
        self.roundTime = 0
        self.length = 0
        


if __name__ == '__main__':
    g = Game(FPS=60, stepTime='FPS')
    #g.loadGame()
    g.runByClick(save=True)
