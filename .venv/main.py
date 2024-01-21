# packages importing
import pygame
import numpy as np
import cv2
import mediapipe as mp
import random
import math


# initializing pygame window
window_height = 650
window_width = 920
fps = 30

white = (255, 255, 255)
green = (0, 255, 0)

pygame.init()
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("My Game")
clock = pygame.time.Clock()


# initializing fundamental geometry classes
class Point:
    def __init__(self, i1, i2=None, polar=False):
        self.y = None
        self.x = None
        if isinstance(i1, Point):
            self.x = i1.x
            self.y = i1.y
        if not polar:
            self.x = i1
            self.y = i2
        else:
            self.x = math.cos(i2) * i1
            self.y = math.sin(i2) * i1

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __str__(self):
        return str((self.x, self.y))

    def dist(self, p=None, y=None):
        if p is None:
            return abs(self)
        if y is not None:
            return math.hypot(abs(p - self.x), abs(y - self.y))
        elif isinstance(p, Point):
            return math.hypot(abs(p.x - self.x), abs(p.y - self.y))


class Vector(Point):
    def __init__(self, i1, i2=None, i3=None, i4=None):
        if isinstance(i1, Vector):
            super().__init__(i1.x, i1.y)
        elif isinstance(i1, Point):
            if isinstance(i2, Point):
                super().__init__(i2.x - i1.x, i2.y - i1.y)
            else:
                super().__init__(i1.x, i1.y)
        elif (isinstance(i1, int) or isinstance(i1, float)) and (isinstance(i2, int) or isinstance(i2, float)):
            if (isinstance(i3, int) or isinstance(i3, float)) and (isinstance(i4, int) or isinstance(i4, float)):
                super().__init__(i3 - i1, i4 - i2)
            else:
                super().__init__(i1, i2)

    def length(self):
        return math.hypot(self.x, self.y) + 1

    def angle(self):
        return math.acos(self.x / self.length())

    def dot_product(self, v2):
        return self.x * v2.x + self.y * v2.y

    def __mul__(self, v2):
        return self.x * v2.x + self.y * v2.y

    def cross_product(self, v2):
        return self.x * v2.y - self.y * v2.x

    def __xor__(self, v2):
        return self.x * v2.y - self.y * v2.x

    def mul(self, n):
        return Vector(self.x * n, self.y * n)

    def __rmul__(self, n):
        return Vector(self.x * n, self.y * n)


# initializing function for image format conversion
def cvimage_to_pygame(image, BGRA=False):
    if BGRA:
        return pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "BGRA")
    else:
        return pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "BGR")

# initializing sprite classes
all_sprites = pygame.sprite.Group()


class PlayerBody(pygame.sprite.Sprite):  # static parts of the player

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('sprites\\player\\legs+body+other_arm+head.png')
        self.image.set_colorkey((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.center = (window_width / 2, window_height / 2)
        self.layer = 2

    def update(self, *args, **kwargs): # movement control
        keystate = pygame.key.get_pressed()
        if keystate[pygame.K_a] and self.rect.left >= 3:
            self.rect.x -= window_width / 210
        if keystate[pygame.K_d] and self.rect.right <= 917:
            self.rect.x += window_width / 210
        if keystate[pygame.K_w] and self.rect.top >= 3:
            self.rect.y -= window_width / 210
        if keystate[pygame.K_s] and self.rect.bottom <= 647:
            self.rect.y += window_width / 210


class PlayerWandArm(pygame.sprite.Sprite):  # the left arm of the player

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
                                   ]).reshape((7, 7))  # array of possible arm sprites for different hand positions
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
        # x_tip, x_center, border, y_tip and y_center declared lower
        # choosing position and sprite of the hand
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


wand_projectile_cv = cv2.imread('sprites\\projectiles\\wand_projectile3.png')


class WandProjectile(pygame.sprite.Sprite):  # an element of the wand trail
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        image_BGR = wand_projectile_cv.copy()
        image_BGRA = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2BGRA)
        image_BGRA[:, :, 3] = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)[:, :, 2]
        self.image = cvimage_to_pygame(image_BGRA, True)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.layer = 3
        self.lifetime = 30
        self.in_line = False

    def update(self):
        self.lifetime -= 1
        self.image.set_alpha(int(self.lifetime / 30 * 255))
        if self.lifetime <= 0:
            self.kill()
        if self.in_line:  # changing the color based on wand trail length and shape
            image_BGR = wand_projectile_cv.copy()
            ret, image_BGR[:, :, 2] = cv2.threshold(image_BGR[:, :, 2], 1, line_color[0], cv2.THRESH_BINARY)
            ret, image_BGR[:, :, 1] = cv2.threshold(image_BGR[:, :, 1], 1, line_color[1], cv2.THRESH_BINARY)
            ret, image_BGR[:, :, 0] = cv2.threshold(image_BGR[:, :, 0], 1, line_color[2], cv2.THRESH_BINARY)
            image_BGRA = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2BGRA)
            # making the less bright pixels transparent
            image_BGRA[:, :, 3] = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)[:, :, 2]
            self.image = cvimage_to_pygame(image_BGRA, True)
            if kill_line:  # reseting the color
                self.in_line = False
                image_BGR = wand_projectile_cv.copy()
                image_BGRA = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2BGRA)
                image_BGRA[:, :, 3] = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)[:, :, 2]
                self.image = cvimage_to_pygame(image_BGRA, True)


class EnemyHealthSign(pygame.sprite.Sprite):  # a part of an enemy's health bar, has possible 4 types
    def __init__(self, type, parent, pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('sprites\\enemies\\health_bar\\'
                                       + ['vertical.png', 'horizontal.png', 'diagonal_upper.png',
                                          'diagonal_lower.png'][type])
        self.image.set_colorkey((255, 255, 255))
        self.rect = self.image.get_rect()
        self.type = type
        self.parent = parent
        self.pos = pos

    def update(self, *args, **kwargs):
        self.rect.x = self.parent.rect.x + self.pos[0]
        self.rect.y = self.parent.rect.y + self.pos[1]


class AbstractEnemy(pygame.sprite.Sprite):  # a parent class for all enemies
    def __init__(self, speed, enemy_score, sign_types):
        pygame.sprite.Sprite.__init__(self)
        self.speed = speed
        self.score = enemy_score  # how much points will the enemy give on death
        self.health = len(sign_types)
        self.pos = random.randrange(4)
        self.signs = []
        global all_sprites
        for i in range(len(sign_types)):  # creating the health bar
            new_sign = EnemyHealthSign(sign_types[i], self, (18 * i, -20))
            self.signs.append(new_sign)
            all_sprites.add(new_sign)
    def update(self):
        # moving towards the player
        if self.rect.x > player_body.rect.x:
            self.rect.x -= self.speed
        elif self.rect.x < player_body.rect.x:
            self.rect.x += self.speed
        if self.rect.y > player_body.rect.y:
            self.rect.y -= self.speed
        elif self.rect.x < player_body.rect.x:
            self.rect.y += self.speed
        # destroying itself and lowering player's health when colliding with the player
        if (pygame.Rect.colliderect(self.rect, player_body.rect) or
            pygame.Rect.colliderect(self.rect, player_wand_arm.rect)):
            global health, score, enemy_counter
            hearts[lives - health].die()
            health -= 1
            for sign in self.signs:
                sign.kill()
            enemy_counter -= 1
            self.kill()
        # lowering the health and updating the healthbar when the player draws the correct symbol
        if line_status == self.signs[-1].type:
            self.signs[-1].kill()
            del self.signs[-1]
            self.health -= 1
            if self.health == 0:
                score += self.score
                enemy_counter -= 1
                self.kill()


class Ghost1(AbstractEnemy):  # a big and slow ghost
    def __init__(self, sign_types):
        AbstractEnemy.__init__(self, 5, 150, sign_types)
        self.image = pygame.image.load('sprites\\enemies\\enemies\\ghost1.jpg')
        self.image.set_colorkey((255, 255, 255))
        self.rect = self.image.get_rect()
        pos = random.randrange(3)
        if pos == 0:
            self.rect.x = -120
            self.rect.y = random.randrange(window_height)
        elif pos == 1:
            self.rect.x = window_width + 120
            self.rect.y = random.randrange(window_width)
        elif pos == 2:
            self.rect.y = window_height + 150
            self.rect.x = random.randrange(window_width)


class Ghost2(AbstractEnemy): # a small and fast ghost
    def __init__(self, sign_types):
        AbstractEnemy.__init__(self, 15, 100, sign_types)
        self.image = pygame.image.load('sprites\\enemies\\enemies\\ghost2.png')
        self.image.set_colorkey((255, 255, 255))
        self.rect = self.image.get_rect()
        pos = random.randrange(3)
        if pos == 0:
            self.rect.x = -60
            self.rect.y = random.randrange(window_height)
        elif pos == 1:
            self.rect.x = window_width + 60
            self.rect.y = random.randrange(window_width)
        elif pos == 2:
            self.rect.y = window_height + 75
            self.rect.x = random.randrange(window_width)


class Heart(pygame.sprite.Sprite):  # class used for displaying the healthbar
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = cv2.imread('sprites\\interface\\heart_full.jpg')
        dim = (int(self.image.shape[1] * 0.5), int(self.image.shape[0] * 0.5))
        self.image = cvimage_to_pygame(cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA))
        self.image.set_colorkey((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.layer = 1

    def die(self):
        self.image = cv2.imread('sprites\\interface\\heart_dead.jpg')
        dim = (int(self.image.shape[1] * 0.5), int(self.image.shape[0] * 0.5))
        self.image = cvimage_to_pygame(cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA))
        self.image.set_colorkey((255, 255, 255))


class Video(pygame.sprite.Sprite):  # class used for displaying video
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


# finding the available cams
dev_port = 0
is_working = True
available_cams = []
while is_working:
    camera = cv2.VideoCapture(dev_port)
    if not camera.isOpened():
        is_working = False
    else:
        is_reading, img = camera.read()
        if is_reading:
            available_cams.append(dev_port)
    dev_port += 1


# hands detector and video display window initializing
handsDetector = mp.solutions.hands.Hands()
cam_index = 0
cap = cv2.VideoCapture(available_cams[cam_index])
video = Video()
all_sprites.add(video)
resize_k = 1
ret, frame = cap.read()
if frame.shape[0] / window_height > frame.shape[1] / window_width:
    resize_k = window_height / frame.shape[0]
else:
    resize_k = window_width / frame.shape[1]
running = True
# letting player choose the optimal drawing area
while running:
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *  # x coordinate of the fingertip
                    flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *  # y coordinate of the fingertip
                    flippedRGB.shape[0])
        x_base = int(results.multi_hand_landmarks[0].landmark[5].x *  # x coordinate of base of the finger
                     flippedRGB.shape[1])
        y_base = int(results.multi_hand_landmarks[0].landmark[5].y *  # y coordinate of base of the finger
                     flippedRGB.shape[0])
        # a measure used for dividing the drawing area into 49 squares
        border = (abs(x_base - x_tip) ** 2 + abs(y_base - y_tip) ** 2) ** 0.5 * 0.5
        x_center = x_tip
        y_center = y_tip
        x_left_max = int(x_center - border * 3.5)
        y_top_max = int(y_center - border * 3.5)
        x_right_max = int(x_center + border * 3.5)
        y_bottom_max = int(y_center + border * 3.5)
        # visualizing the drawing area
        cv2.line(flippedRGB, (x_left_max, y_top_max), (x_right_max, y_top_max), (0, 255, 0), 5)
        cv2.line(flippedRGB, (x_right_max, y_top_max), (x_right_max, y_bottom_max), (0, 255, 0), 5)
        cv2.line(flippedRGB, (x_right_max, y_bottom_max), (x_left_max, y_bottom_max), (0, 255, 0), 5)
        cv2.line(flippedRGB, (x_left_max, y_bottom_max), (x_left_max, y_top_max), (0, 255, 0), 5)
        cv2.circle(flippedRGB, (x_center, y_center), 10, (0, 0, 255), -1)
        cv2.circle(flippedRGB, (x_tip, y_tip), 10, (255, 0, 0), -1)
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and results is not None:
            if event.key == pygame.K_c:  # changing he camera
                cam_index += 1
                if cam_index > len(available_cams) - 1:
                    cam_index = 0
                cap = cv2.VideoCapture(available_cams[cam_index])
                ret, frame = cap.read()
                if frame.shape[0] / window_height > frame.shape[1] / window_width:
                    resize_k = window_height / frame.shape[0]
                else:
                    resize_k = window_width / frame.shape[1]
            if event.key == pygame.K_SPACE and results is not None:  # locking the drawing area and starting the game
                x_base = int(results.multi_hand_landmarks[0].landmark[5].x *
                             flippedRGB.shape[1])
                y_base = int(results.multi_hand_landmarks[0].landmark[5].y *
                             flippedRGB.shape[0])
                border = (abs(x_base - x_tip) ** 2 + abs(y_base - y_tip) ** 2) ** 0.5 * 0.5
                x_center = x_tip
                y_center = y_tip
                x_left_max = int(x_center - border * 3.5)
                y_top_max = int(y_center - border * 3.5)
                x_right_max = int(x_center + border * 3.5)
                y_bottom_max = int(y_center + border * 3.5)
                running = False
    # rendering
    screen.fill(white)
    all_sprites.draw(screen)
    dim = (int(res_image.shape[1] * resize_k), int(res_image.shape[0] * resize_k))
    video.display(cv2.resize(res_image, dim, interpolation=cv2.INTER_AREA))
    pygame.display.flip()
video.kill()

# initializing various variables
wand_pos_prev = Point(512, 248)
wand_vector_prev = Vector(wand_pos_prev, wand_pos_prev)
wand_projectiles_prev = []
new_projectile = WandProjectile(512, 248)
wand_delta_x_prev_positive = True
line_color = (0, 0, 0)
kill_line = False
results = None
line_length = 0
horizon = Vector(1, 1, 100, 1)
line_angle = 0
line_angle_prev = 0
enemy_counter = 0
line_length_max = 10
line_sector_prev = 0
player_body = PlayerBody()
all_sprites.add(player_body)
player_wand_arm = PlayerWandArm()
all_sprites.add(player_wand_arm)
show_video = False
running = True
level = 0
score = 0
# creating the healthbar
lives = 7
health = lives
hearts = []
for i in range(lives):
    new_heart = Heart(770 - 70 * i, 30)
    hearts.append(new_heart)
    all_sprites.add(new_heart)
# the game
while running:
    clock.tick(fps)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:  # opening/closing the video
                if show_video:
                    show_video = False
                    video.kill()
                else:
                    show_video = True
                    all_sprites.add(video)
            elif event.key == pygame.K_c:  # changing the camera
                cam_index += 1
                if cam_index > len(available_cams) - 1:
                    cam_index = 0
                cap = cv2.VideoCapture(available_cams[cam_index])
            elif event.key == pygame.K_SPACE and results is not None:  # resetting the arm and the drawing area
                x_base = int(results.multi_hand_landmarks[0].landmark[5].x *
                            flippedRGB.shape[1])
                y_base = int(results.multi_hand_landmarks[0].landmark[5].y *
                            flippedRGB.shape[0])
                border = (abs(x_base - x_tip) ** 2 + abs(y_base - y_tip) ** 2) ** 0.5 * 0.5
                x_center = x_tip
                y_center = y_tip
                x_left_max = int(x_center - border * 3.5)
                y_top_max = int(y_center - border * 3.5)
                x_right_max = int(x_center + border * 3.5)
                y_bottom_max = int(y_center + border * 3.5)
                player_wand_arm.x_area = 4
                player_wand_arm.y_area = 4

    # hand detection and processing
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                    flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                    flippedRGB.shape[0])
        cv2.line(flippedRGB, (x_left_max, y_top_max), (x_right_max, y_top_max), (0, 255, 0), 5)
        cv2.line(flippedRGB, (x_right_max, y_top_max), (x_right_max, y_bottom_max), (0, 255, 0), 5)
        cv2.line(flippedRGB, (x_right_max, y_bottom_max), (x_left_max, y_bottom_max), (0, 255, 0), 5)
        cv2.line(flippedRGB, (x_left_max, y_bottom_max), (x_left_max, y_top_max), (0, 255, 0), 5)
        cv2.circle(flippedRGB, (x_center, y_center), 10, (0, 0, 255), -1)
        cv2.circle(flippedRGB, (x_tip, y_tip), 10, (255, 0, 0), -1)
    if show_video:
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        dim = (int(window_width / 4), int(window_height / 4))
        video.display(cv2.resize(res_image, dim, interpolation=cv2.INTER_AREA))

    kill_line = False
    line_status = -1
    new_wand_projectiles = []
    # calculating the position of the wand tip
    wand_pos = Point(0, 0)
    if x_tip <= x_left_max:
        wand_pos.x = player_body.rect.x + 59
    elif x_tip >= x_right_max:
        wand_pos.x = player_body.rect.x + 179
    else:
        wand_pos.x = int((x_tip - x_left_max) / (border * 7) * 120 + player_body.rect.x + 59)
    if y_tip <= y_top_max:
        wand_pos.y = player_body.rect.y - 45
    elif y_tip >= y_bottom_max:
        wand_pos.y = player_body.rect.y + 75
    else:
        wand_pos.y = int((y_tip - y_top_max) / (border * 7) * 120 + player_body.rect.y - 45)
    # analyzing the wand movement
    wand_delta_x = abs(wand_pos.x - wand_pos_prev.x) + 1
    wand_delta_y = abs(wand_pos.y - wand_pos_prev.y) + 1
    wand_delta_x2 = wand_pos.x - wand_pos_prev.x
    wand_delta_y2 = wand_pos.y - wand_pos_prev.y
    wand_delta_x_positive = wand_pos.x - wand_pos_prev.x >= 0
    wand_delta_y_positive = wand_pos.y - wand_pos_prev.y >= 0
    wand_vector = Vector(wand_pos_prev, wand_pos)
    line_angle = math.acos((wand_vector * horizon) / (wand_vector.length() * horizon.length())) * 57
    if line_angle < 30 or line_angle > 200:
        line_sector = 0
    elif line_angle < 60 or line_angle > 120:
        if ((horizon ^ wand_vector > 0 and horizon * wand_vector > 0) or
                (horizon ^ wand_vector < 0 and horizon * wand_vector < 0)):
            line_sector = 2
        else:
            line_sector = 3
    else:
        line_sector = 1
    # drawing the wand projectiles
    steps = int(math.hypot(wand_delta_x, wand_delta_y) / 7)
    for i in range(steps):
        new_projectile_x = wand_pos_prev.x + (wand_delta_x2 / steps) * i
        new_projectile_y = wand_pos_prev.y + (wand_delta_y2 / steps) * i
        new_projectile = WandProjectile(new_projectile_x, new_projectile_y)
        all_sprites.add(new_projectile)
        new_wand_projectiles.append(new_projectile)
    new_projectile = WandProjectile(wand_pos.x, wand_pos.y)
    all_sprites.add(new_projectile)
    new_wand_projectiles.append(new_projectile)
    # changing wand trail's color depending on it's angle if the trail is straight enough
    if wand_delta_x_positive == wand_delta_x_prev_positive and line_sector == line_sector_prev:
        for element in wand_projectiles_prev:
            element.in_line = True
            line_length += 1
            if line_sector == 0:
                if line_length < line_length_max:
                    line_color = (0, 0, int(200 / line_length_max) * line_length)
                else:
                    line_color = (0, 0, 200)
            elif line_sector == 2:
                if line_length < line_length_max:
                    line_color = (0, int(200 / line_length_max) * line_length, 0)
                else:
                    line_color = (0, 200, 0)
            elif line_sector == 3:
                if line_length < line_length_max:
                    line_color = (int(255 / line_length_max) * line_length, int(255 / line_length_max) * line_length, 0)
                else:
                    line_color = (255, 255, 0)
            elif line_sector == 1:
                if line_length < line_length_max:
                    line_color = (int(200 / line_length_max) * line_length, 0, 0)
                else:
                    line_color = (200, 0, 0)
        for element in new_wand_projectiles:
            element.in_line = True
            line_length += 1
            if line_sector == 0:
                if line_length < line_length_max:
                    line_color = (0, 0, int(200 / line_length_max) * line_length)
                else:
                    line_color = (0, 0, 200)
            elif line_sector == 2:
                if line_length < line_length_max:
                    line_color = (0, int(200 / line_length_max) * line_length, 0)
                else:
                    line_color = (0, 200, 0)
            elif line_sector == 3:
                if line_length < line_length_max:
                    line_color = (int(255 / line_length_max) * line_length, int(255 / line_length_max) * line_length, 0)
                else:
                    line_color = (255, 255, 0)
            elif line_sector == 1:
                if line_length < line_length_max:
                    line_color = (int(200 / line_length_max) * line_length, 0, 0)
                else:
                    line_color = (200, 0, 0)
    else:
        # resetting the projectiles' color if the trail is not straight
        kill_line = True
        line_length = 0
        line_avg_angle = 0
        line_angles = [0]
    if line_length >= line_length_max:  # signalizing enemy objects if the wand trail is straght and long enough
        line_status = line_sector
    wand_delta_x_prev_positive = wand_delta_x_positive
    wand_pos_prev = wand_pos
    wand_vector_prev = wand_vector
    if line_status == -1:
        wand_projectiles_prev = new_wand_projectiles
    else:
        wand_projectiles_prev = []
    line_angle_prev = line_angle
    line_sector_prev = line_sector

    # spawning enemies depending on the level, increasing the level if the score is high enough
    if level == 0 and enemy_counter <= 3:
        if random.randrange(50) == 0 or enemy_counter == 0:
            all_sprites.add(Ghost1([random.randrange(2)]))
            enemy_counter += 1
            if score >= 450:
                level += 1
    elif level == 1 and enemy_counter <= 3:
        if random.randrange(100) == 0 or enemy_counter == 0:
            all_sprites.add(Ghost1([random.randrange(2) for _ in range(4)]))
            enemy_counter += 1
            if score >= 1200:
                level += 1
    elif level == 2 and enemy_counter <= 3:
        if random.randrange(100) == 0 or enemy_counter == 0:
            all_sprites.add(Ghost1([random.randrange(4) for _ in range(4)]))
            enemy_counter += 1
            if score >= 2000:
                level += 1
    elif level == 3 and enemy_counter <= 5:
        if random.randrange(100) == 0 or enemy_counter == 0:
            all_sprites.add(Ghost1([random.randrange(4) for _ in range(4)]))
            enemy_counter += 1
        if random.randrange(50) == 0 or enemy_counter == 0:
            all_sprites.add(Ghost2([random.randrange(4)]))
            enemy_counter += 1
    # closing the game if player health is zero
    if health <= 0:
        print('Game Over')
        print('Your score:', score)
        break

    # rendering
    screen.fill(white)
    all_sprites.draw(screen)
    all_sprites.update()
    pygame.display.flip()
pygame.quit()
