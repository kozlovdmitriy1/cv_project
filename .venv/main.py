import pygame
import numpy as np
import cv2
import mediapipe as mp
import random
import math


window_height = 650
window_width = 920
fps = 30

white = (255, 255, 255)
green = (0, 255, 0)

pygame.init()
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("My Game")
clock = pygame.time.Clock()

all_sprites = pygame.sprite.Group()
class Player_body(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('sprites\player\legs+body+other_arm+head.png')
        self.rect = self.image.get_rect()
        self.rect.center = (window_width / 2, window_height / 2)

    def update(self, *args, **kwargs):
        keystate = pygame.key.get_pressed()
        if keystate[pygame.K_a] and self.rect.left >= 3:
            self.rect.x -= window_width / 210
        if keystate[pygame.K_d] and self.rect.right <= 917:
            self.rect.x += window_width / 210
        if keystate[pygame.K_w] and self.rect.top >= 3:
            self.rect.y -= window_width / 210
        if keystate[pygame.K_s] and self.rect.bottom <= 647:
            self.rect.y += window_width / 210
class Player_wand_arm(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.positions = np.array([pygame.image.load('sprites\player\wand_arm_1x1.png'),
                                   pygame.image.load('sprites\player\wand_arm_1x2.png'),
                                   pygame.image.load('sprites\player\wand_arm_1x3.png'),
                                   pygame.image.load('sprites\player\wand_arm_1x4.png'),
                                   pygame.image.load('sprites\player\wand_arm_1x5.png'),
                                   pygame.image.load('sprites\player\wand_arm_1x6.png'),
                                   pygame.image.load('sprites\player\wand_arm_1x7.png'),
                                   pygame.image.load('sprites\player\wand_arm_2x1.png'),
                                   pygame.image.load('sprites\player\wand_arm_2x2.png'),
                                   pygame.image.load('sprites\player\wand_arm_2x3.png'),
                                   pygame.image.load('sprites\player\wand_arm_2x4.png'),
                                   pygame.image.load('sprites\player\wand_arm_2x5.png'),
                                   pygame.image.load('sprites\player\wand_arm_2x6.png'),
                                   pygame.image.load('sprites\player\wand_arm_2x7.png'),
                                   pygame.image.load('sprites\player\wand_arm_3x1.png'),
                                   pygame.image.load('sprites\player\wand_arm_3x2.png'),
                                   pygame.image.load('sprites\player\wand_arm_3x3.png'),
                                   pygame.image.load('sprites\player\wand_arm_3x4.png'),
                                   pygame.image.load('sprites\player\wand_arm_3x5.png'),
                                   pygame.image.load('sprites\player\wand_arm_3x6.png'),
                                   pygame.image.load('sprites\player\wand_arm_3x7.png'),
                                   pygame.image.load('sprites\player\wand_arm_4x1.png'),
                                   pygame.image.load('sprites\player\wand_arm_4x2.png'),
                                   pygame.image.load('sprites\player\wand_arm_4x3.png'),
                                   pygame.image.load('sprites\player\wand_arm_4x4.png'),
                                   pygame.image.load('sprites\player\wand_arm_4x5.png'),
                                   pygame.image.load('sprites\player\wand_arm_4x6.png'),
                                   pygame.image.load('sprites\player\wand_arm_4x7.png'),
                                   pygame.image.load('sprites\player\wand_arm_5x1.png'),
                                   pygame.image.load('sprites\player\wand_arm_5x2.png'),
                                   pygame.image.load('sprites\player\wand_arm_5x3.png'),
                                   pygame.image.load('sprites\player\wand_arm_5x4.png'),
                                   pygame.image.load('sprites\player\wand_arm_5x5.png'),
                                   pygame.image.load('sprites\player\wand_arm_5x6.png'),
                                   pygame.image.load('sprites\player\wand_arm_5x7.png'),
                                   pygame.image.load('sprites\player\wand_arm_6x1.png'),
                                   pygame.image.load('sprites\player\wand_arm_6x2.png'),
                                   pygame.image.load('sprites\player\wand_arm_6x3.png'),
                                   pygame.image.load('sprites\player\wand_arm_6x4.png'),
                                   pygame.image.load('sprites\player\wand_arm_6x5.png'),
                                   pygame.image.load('sprites\player\wand_arm_6x6.png'),
                                   pygame.image.load('sprites\player\wand_arm_6x7.png'),
                                   pygame.image.load('sprites\player\wand_arm_7x1.png'),
                                   pygame.image.load('sprites\player\wand_arm_7x2.png'),
                                   pygame.image.load('sprites\player\wand_arm_7x3.png'),
                                   pygame.image.load('sprites\player\wand_arm_7x4.png'),
                                   pygame.image.load('sprites\player\wand_arm_7x5.png'),
                                   pygame.image.load('sprites\player\wand_arm_7x6.png'),
                                   pygame.image.load('sprites\player\wand_arm_7x7.png'),
                                   ]).reshape((7, 7))
        self.y_adjustment = np.array([0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0,
                                      9, 8, 10, 1, 0, 0, 0,
                                      26, 17, 23, 26, 28, 5, 0,
                                      27, 27, 22, 35, 25, 0, 0,
                                      41, 30, 27, 36, 43, 26, 6]).reshape((7, 7))
        self.image = pygame.image.load('sprites\player\wand_arm_4x4.png')
        self.rect = self.image.get_rect()
        self.rect.center = (window_width / 2 + 52, window_height / 2 - 77 + self.y_adjustment[(3, 3)])
        self.x_area = 4
        self.y_area = 4

    def update(self, *args, **kwargs):
        if x_tip - x_center < border * -3.5:
            self.x_area = 1
        elif x_tip - x_center >= border * 3.5:
            self.x_area = 7
        else:
            if x_tip - x_center >= 0:
                self.x_area = math.ceil((x_tip - x_center + 0.5 * border) / border) + 3
            else:
                self.x_area = 5 - math.ceil(abs(x_tip - x_center - 0.5 * border) / border)
        if y_tip - y_center < border * -3.5:
            self.y_area = 1
        elif y_tip - y_center >= border * 3.5:
            self.y_area = 7
        else:
            if y_tip - y_center >= 0:
                self.y_area = math.ceil((y_tip - y_center + 0.5 * border) / border) + 3
            else:
                self.y_area = 5 - math.ceil(abs(y_tip - y_center - 0.5 * border) / border)
        self.image = self.positions[(self.y_area - 1, self.x_area - 1)]
        self.rect.x = player_body.rect.x + 59
        self.rect.y = player_body.rect.y - 45 + self.y_adjustment[(self.y_area - 1, self.x_area - 1)]

player_body = Player_body()
all_sprites.add(player_body)
player_wand_arm = Player_wand_arm()
all_sprites.add(player_wand_arm)

dev_port = 0
is_working = True
avalible_cams = []
while is_working:
    camera = cv2.VideoCapture(dev_port)
    if not camera.isOpened():
        is_working = False
    else:
        is_reading, img = camera.read()
        if is_reading:
            avalible_cams.append(dev_port)
    dev_port += 1
handsDetector = mp.solutions.hands.Hands()
cam_index = 0
cap = cv2.VideoCapture(avalible_cams[cam_index])
show_video = False
running = True
while running:
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                    flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                    flippedRGB.shape[0])
        cv2.circle(flippedRGB, (x_tip, y_tip), 10, (255, 0, 0), -1)
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and results is not None:
            if event.key == pygame.K_v:
                if show_video:
                    show_video = False
                    cv2.destroyAllWindows()
                else:
                    show_video = True
            elif event.key == pygame.K_c:
                cam_index += 1
                if cam_index > len(avalible_cams) - 1:
                    cam_index = 0
                cap = cv2.VideoCapture(avalible_cams[cam_index])
            if event.key == pygame.K_SPACE:
                x_base = int(results.multi_hand_landmarks[0].landmark[5].x *
                             flippedRGB.shape[1])
                y_base = int(results.multi_hand_landmarks[0].landmark[5].y *
                             flippedRGB.shape[0])
                border = (abs(x_base - x_tip) ** 2 + abs(y_base - y_tip) ** 2) ** 0.5
                x_center = x_tip
                y_center = y_tip
                player_wand_arm.x_area = 4
                player_wand_arm.y_area = 4
                running = False
                cv2.destroyAllWindows()

running = True
while running:
    clock.tick(fps)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:
                if show_video:
                    show_video = False
                    cv2.destroyAllWindows()
                else:
                    show_video = True
            elif event.key == pygame.K_c:
                cam_index += 1
                if cam_index > len(avalible_cams) - 1:
                    cam_index = 0
                cap = cv2.VideoCapture(avalible_cams[cam_index])
            elif event.key == pygame.K_SPACE:
                x_base = int(results.multi_hand_landmarks[0].landmark[5].x *
                            flippedRGB.shape[1])
                y_base = int(results.multi_hand_landmarks[0].landmark[5].y *
                            flippedRGB.shape[0])
                border = (abs(x_base - x_tip) ** 2 + abs(y_base - y_tip) ** 2) ** 0.5
                x_center = x_tip
                y_center = y_tip
                player_wand_arm.x_area = 4
                player_wand_arm.y_area = 4

    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                    flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                    flippedRGB.shape[0])
        cv2.circle(flippedRGB, (x_tip, y_tip), 10, (255, 0, 0), -1)
    if show_video:
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hands", res_image)

    screen.fill(white)
    all_sprites.draw(screen)
    all_sprites.update()
    pygame.display.flip()
pygame.quit()
