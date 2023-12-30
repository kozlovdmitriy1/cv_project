import pygame
import numpy as np
import cv2
import mediapipe
import random


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
class Player(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((window_width / 18, window_width / 18))
        self.image.fill(green)
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

player = Player()
all_sprites.add(player)

running = True
while running:
    clock.tick(fps)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(white)
    all_sprites.draw(screen)
    all_sprites.update()
    pygame.display.flip()

pygame.quit()
