import cv2
import pygame
import sys
import numpy as np
import random
from pygame.locals import *

pygame.init()

def minmax(v):
    if v > 255:
        v = 255
    if v < 0:
        v = 0
    return v


def dither(image, samplingF=1):
    h = image.shape[0]
    w = image.shape[1]
    
    for y in range(h-1):
        for x in range(1, w-1):
            old_p = image[y, x]
            new_p = np.round(samplingF * old_p/255.0) * (255/samplingF)
            image[y, x] = new_p
            
            quant_error_p = old_p - new_p
                        
            image[y, x+1] = minmax(image[y, x+1] + quant_error_p * 7 / 16.0)
            image[y+1, x-1] = minmax(image[y+1, x-1] + quant_error_p * 3 / 16.0)
            image[y+1, x] = minmax(image[y+1, x] + quant_error_p * 5 / 16.0)
            image[y+1, x+1] = minmax(image[y+1, x+1] + quant_error_p * 1 / 16.0)

    return (np.round(image/255)*255).astype("uint8")


ds=3

image = cv2.imread(sys.argv[1])
image = cv2.resize(image, (0,0), fx=1/ds, fy=1/ds) 
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

dithered = dither(image)

scale = 5
height, width = image.shape

screen = pygame.Surface((width*scale, height*scale))

pygame.display.set_caption("Pins")

linkCount = 7

class Node:
    def __init__(self, x, y, depth):
        self.x = x
        self.y = y
        self.depth = depth
        self.closest = []
        self.linked = []

nodes = []
nodeMap = []

nodeDist = 1

for y in range(height):
    nodeMap.append([])
    for x in range(width):
        if dithered[y, x] == 0:
            top = max(0, y-nodeDist)
            left = max(0, x-nodeDist)

            valid = True
            for checkY in range(top, y+1):
                for checkX in range(left, x+1):
                    if (checkY != y or checkX != x) and nodeMap[checkY][checkX] is not None:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                bottom = min(height, y+nodeDist+1)
                right = min(width, x+nodeDist+1)

                depth = 0
                for depthY in range(top, bottom):
                    for depthX in range(left, right):
                        if dithered[depthY, depthX] == 0:
                            depth += 1
                
                node = Node(x, y, max(1,min(3,depth//((nodeDist**2)*2))))
                nodes.append(node)
                nodeMap[-1].append(node)
            else:
                nodeMap[-1].append(None)
        else:
            nodeMap[-1].append(None)


for node in nodes:
    r = 1
    while (node.x-r >= 0 or node.x+r < width or node.y-r >= 0 or node.y+r <height) and len(node.closest) < linkCount:
        top = max(0, node.y-r)
        bottom = min(height, node.y+r+1)
        left = max(0, node.x-r)
        right =min(width, node.x+r+1)
        
        section = dithered[top:bottom, left:right]
        closest = []
        secHeight, secWidth = section.shape
        for y in range(secHeight):
                for x in range(secWidth):
                    if nodeMap[top+y][left+x] is not None and (x != node.x or y != node.y):
                        closest.append(nodeMap[top+y][left+x])
        if len(closest) >= linkCount:
            random.shuffle(closest)
            node.closest = closest
            break
        
        r += 1
    
def get_pos(x, y, w=1):
    return (round(x * scale + (scale-w)/2), round(y * scale+ (scale-w)/2))

endNode = nodes[0]

frameCount = 0

screen.fill((255, 255, 255))
for node in nodes:
    pygame.draw.circle(screen, (200, 200, 200), get_pos(node.x, node.y), 3)
    pygame.draw.circle(screen, (100, 100, 100), get_pos(node.x, node.y), 3, 1)

oldEndNode = None

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("video.mp4", fourcc, 20.0, (width*scale, height*scale))

while True:
    frameName = "frames/{}.png".format(frameCount)
    oldEndNode = endNode

    if frameCount > 0:
        needToSearch = True
        for closest in endNode.closest:
            if endNode.linked.count(closest) < max(endNode.depth, closest.depth):
                endNode.linked.append(closest)
                closest.linked.append(endNode)
                endNode = closest
                needToSearch = False
                break
        if needToSearch:
            found = False
            for node in nodes:
                for closest in node.closest:
                    if node.linked.count(closest) < max(node.depth, closest.depth):
                        endNode.linked.append(node)
                        node.linked.append(endNode)
                        endNode = node
                        found = True
                        break
                if found:
                    break
            if not found:
                pygame.image.save(screen, frameName)
                frame = cv2.imread(frameName)
                out.write(frame)
                break

    pygame.draw.line(screen, (0, 0, 0), get_pos(oldEndNode.x, oldEndNode.y), get_pos(endNode.x, endNode.y), oldEndNode.linked.count(endNode))

    pygame.draw.circle(screen, (200, 200, 200), get_pos(oldEndNode.x, oldEndNode.y), 3)
    pygame.draw.circle(screen, (100, 100, 100), get_pos(oldEndNode.x, oldEndNode.y), 3, 1)

    pygame.draw.circle(screen, (255, 0, 0), get_pos(endNode.x, endNode.y), 3)
    pygame.draw.circle(screen, (100, 100, 100), get_pos(endNode.x, endNode.y), 3, 1)
        
    pygame.image.save(screen, frameName)
    frame = cv2.imread(frameName)
    out.write(frame)

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
    
    frameCount += 1

print("Releasing")
out.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()
