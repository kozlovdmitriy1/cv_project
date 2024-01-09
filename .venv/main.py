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


def cvimage_to_pygame(image):
    return pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "BGR")


all_sprites = pygame.sprite.Group()


class PlayerBody(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('sprites\\player\\legs+body+other_arm+head.png')
        self.image.set_colorkey((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.center = (window_width / 2, window_height / 2)
        self.layer = 2

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


class PlayerWandArm(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.positions = np.array([pygame.image.load('sprites\\player\\wand_arm_1x1.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_1x2.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_1x3.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_1x4.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_1x5.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_1x6.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_1x7.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_2x1.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_2x2.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_2x3.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_2x4.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_2x5.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_2x6.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_2x7.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_3x1.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_3x2.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_3x3.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_3x4.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_3x5.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_3x6.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_3x7.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_4x1.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_4x2.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_4x3.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_4x4.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_4x5.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_4x6.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_4x7.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_5x1.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_5x2.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_5x3.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_5x4.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_5x5.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_5x6.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_5x7.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_6x1.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_6x2.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_6x3.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_6x4.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_6x5.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_6x6.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_6x7.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_7x1.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_7x2.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_7x3.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_7x4.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_7x5.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_7x6.png'),
                                   pygame.image.load('sprites\\player\\wand_arm_7x7.png'),
                                   ]).reshape((7, 7))
        self.y_adjustment = np.array([0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0,
                                      9, 8, 10, 1, 0, 0, 0,
                                      26, 17, 23, 26, 28, 5, 0,
                                      27, 27, 22, 35, 25, 25, 0,
                                      41, 30, 27, 36, 43, 26, 6]).reshape((7, 7))
        self.image = pygame.image.load('sprites\\player\\wand_arm_4x4.png')
        self.image.set_colorkey((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.center = (window_width / 2 + 52, window_height / 2 - 77 + self.y_adjustment[(3, 3)])
        self.x_area = 4
        self.y_area = 4
        self.layer = 2

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


class WandProjectile(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        image_BGR = cv2.imread('sprites\\projectiles\\wand_projectile3.png')
        image_BGRA = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2BGRA)
        image_BGRA[:, :, 3] = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)[:, :, 2]
        self.image = pygame.image.frombuffer(image_BGRA.tobytes(), image_BGRA.shape[1::-1], "BGRA")
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.layer = 3
        self.lifetime = 30

    def update(self):
        self.lifetime -= 1
        self.image.set_alpha(int(self.lifetime / 30 * 255))
        if self.lifetime <= 0:
            self.kill()


class Video(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50, 50))
        self.rect = self.image.get_rect()
        self.rect.center = (window_width / 2, window_height / 2)
        self.layer = 1

    def update(self, *args, **kwargs):
        pass

    def display(self, image):
        self.image = cvimage_to_pygame(image)
        self.rect = self.image.get_rect()


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
video = Video()
all_sprites.add(video)
resize_k = 1
ret, frame = cap.read()
if frame.shape[0] / window_height > frame.shape[1] / window_width:
    resize_k = window_height / frame.shape[0]
else:
    resize_k = window_width / frame.shape[1]
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
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and results is not None:
            if event.key == pygame.K_c:
                cam_index += 1
                if cam_index > len(avalible_cams) - 1:
                    cam_index = 0
                cap = cv2.VideoCapture(avalible_cams[cam_index])
                ret, frame = cap.read()
                if frame.shape[0] / window_height > frame.shape[1] / window_width:
                    resize_k = window_height / frame.shape[0]
                else:
                    resize_k = window_width / frame.shape[1]
            if event.key == pygame.K_SPACE and results is not None:
                x_base = int(results.multi_hand_landmarks[0].landmark[5].x *
                             flippedRGB.shape[1])
                y_base = int(results.multi_hand_landmarks[0].landmark[5].y *
                             flippedRGB.shape[0])
                border = (abs(x_base - x_tip) ** 2 + abs(y_base - y_tip) ** 2) ** 0.5 * 0.75
                x_center = x_tip
                y_center = y_tip
                x_left_top = int(x_center - border * 3.5)
                y_left_top = int(y_center - border * 3.5)
                x_right_bottom = int(x_center + border * 3.5)
                y_right_bottom = int(y_center + border * 3.5)
                running = False
    screen.fill(white)
    all_sprites.draw(screen)
    dim = (int(res_image.shape[1] * resize_k), int(res_image.shape[0] * resize_k))
    video.display(cv2.resize(res_image, dim, interpolation=cv2.INTER_AREA))
    pygame.display.flip()
video.kill()


player_body = PlayerBody()
all_sprites.add(player_body)
player_wand_arm = PlayerWandArm()
all_sprites.add(player_wand_arm)
show_video = False
running = True
x_wand_prev = 512
y_wand_prev = 248
while running:
    clock.tick(fps)
    results = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:
                if show_video:
                    show_video = False
                    video.kill()
                else:
                    show_video = True
                    all_sprites.add(video)
            elif event.key == pygame.K_c:
                cam_index += 1
                if cam_index > len(avalible_cams) - 1:
                    cam_index = 0
                cap = cv2.VideoCapture(avalible_cams[cam_index])
            elif event.key == pygame.K_SPACE and results is not None:
                x_base = int(results.multi_hand_landmarks[0].landmark[5].x *
                            flippedRGB.shape[1])
                y_base = int(results.multi_hand_landmarks[0].landmark[5].y *
                            flippedRGB.shape[0])
                border = (abs(x_base - x_tip) ** 2 + abs(y_base - y_tip) ** 2) ** 0.5 * 0.75
                x_center = x_tip
                y_center = y_tip
                x_left_top = int(x_center - border * 3.5)
                y_left_top = int(y_center - border * 3.5)
                x_right_bottom = int(x_center + border * 3.5)
                y_right_bottom = int(y_center + border * 3.5)
                player_wand_arm.x_area = 4
                player_wand_arm.y_area = 4
                print(x_left_top, y_left_top)
                print(x_right_bottom, y_right_bottom)
                print(frame.shape[1], frame.shape[0])

    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                    flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                    flippedRGB.shape[0])
        cv2.circle(flippedRGB, (x_left_top, y_left_top), 10, (0, 255, 0), -1)
        cv2.circle(flippedRGB, (x_right_bottom, y_right_bottom), 10, (0, 255, 0), -1)
        cv2.circle(flippedRGB, (x_center, y_center), 10, (0, 0, 255), -1)
        cv2.circle(flippedRGB, (x_tip, y_tip), 10, (255, 0, 0), -1)
    if show_video:
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        dim = (int(window_width / 4), int(window_height / 4))
        video.display(cv2.resize(res_image, dim, interpolation=cv2.INTER_AREA))

    if x_tip <= x_left_top:
        x_wand = player_body.rect.x + 59
    elif x_tip >= x_right_bottom:
        x_wand = player_body.rect.x + 179
    else:
        x_wand = int((x_tip - x_left_top) / (border * 7) * 120 + player_body.rect.x + 59)
    if y_tip <= y_left_top:
        y_wand = player_body.rect.y - 45
    elif y_tip >= y_right_bottom:
        y_wand = player_body.rect.y + 75
    else:
        y_wand = int((y_tip - y_left_top) / (border * 7) * 120 + player_body.rect.y - 45)
    wand_delta_x = abs(x_wand - x_wand_prev) + 1
    wand_delta_y = y_wand - y_wand_prev + 1
    sin = wand_delta_y / wand_delta_x
    if sin >= 1:
        wand_prjectile_gap_x = int(7 / abs(sin)) + 1
    else:
        wand_prjectile_gap_x = 7
    if x_wand - x_wand_prev >= 0:
        for i in range(int(wand_delta_x / wand_prjectile_gap_x)):
            all_sprites.add(WandProjectile(x_wand_prev + i * wand_prjectile_gap_x,
                                           y_wand_prev + wand_prjectile_gap_x * sin * i))
        all_sprites.add(WandProjectile(x_wand, y_wand))
    else:
        for i in range(int(wand_delta_x / wand_prjectile_gap_x)):
            all_sprites.add(WandProjectile(x_wand_prev - i * wand_prjectile_gap_x,
                                           y_wand_prev + wand_prjectile_gap_x * sin * i))
    all_sprites.add(WandProjectile(x_wand, y_wand))
    x_wand_prev = x_wand
    y_wand_prev = y_wand
    screen.fill(white)
    all_sprites.draw(screen)
    all_sprites.update()
    pygame.display.flip()
pygame.quit()
