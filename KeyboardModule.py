import pygame

def init():
    pygame.init()
    win = pygame.display.set_mode((100,100))

def getKeyInput():
    """Get keyboard input state for all relevant keys"""
    pygame.event.pump()
    keyList = [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_r, pygame.K_f]
    keys = pygame.key.get_pressed()
    keyInput = {'W': keys[keyList[0]],
                'S': keys[keyList[1]],
                'A': keys[keyList[2]],
                'D': keys[keyList[3]],
                'R': keys[keyList[4]],
                'F': keys[keyList[5]]}  # Added F key
    return keyInput

def main():
    """Test function for keyboard module"""
    print('Press keys to test...')
    print('Press ESC to exit')
    init()
    while True:
        keyInput = getKeyInput()
        print(keyInput, end='\r')
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            break
    pygame.quit()

if __name__ == '__main__':
    main()