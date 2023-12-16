import pytube
import os
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def is_shot_made(current_center, previous_center, basket):
    # Given a current ball position, previous ball position, and a basket data determine if a shot is made
    # y is distance from the top and x is distance from the left
    # examble basket [1043 255 1076 283] x1, y1, x2, y2
    current_center_x, current_center_y = current_center
    previous_center_y = previous_center[1]


    basket_left, basket_top, basket_right, basket_bottom = basket
    #is ball in the net
    is_x_in_basket = basket_left <= current_center_x and current_center_x <= basket_right
    is_y_in_basket = basket_top <= current_center_y and basket_bottom >= current_center_y
    is_ball_decending = previous_center_y < current_center_y

    return (is_x_in_basket and is_y_in_basket and is_ball_decending)


def is_ball_in_basket(current_ball, prev_ball, basket):
    # Alternative to is_shot_made. Testing showed no improvement in accuracy
    # Given a current ball position, previous ball position, and a basket data determine if a shot is made
    # y is distance from the top and x is distance from the left
    # examble basket [1043 255 1076 283] x1, y1, x2, y2
        # adjust_basket = 0
        # adjust ball = 0
        current_left_x, current_top_y, current_right_x, current_bottom_y = current_ball
        previous_bottom_y = prev_ball[1]


        basket_left, basket_top, basket_right, basket_bottom = basket
        #is ball in the net
        is_x_in_basket = basket_left <= current_left_x and current_right_x <= basket_right
        is_y_in_basket = basket_top <= current_top_y and basket_bottom >= current_bottom_y
        is_ball_decending = previous_bottom_y < current_bottom_y
        return (is_x_in_basket and is_y_in_basket and is_ball_decending)


def is_near_basket(basket, ball_center):
    area = 50
    basket_left, basket_top, basket_right, basket_bottom = basket
    within_hoop_y = (basket_top-area) <= ball_center[1] and ball_center[1] <= (basket_bottom+area)
    within_hoop_x = (basket_left-area) <= ball_center[0] and ball_center[0] <= (basket_right+area)
    return within_hoop_x & within_hoop_y


def download_youtube_video(url):
    yt = pytube.YouTube(url)
    stream = yt.streams.get_highest_resolution()
    return stream.download()



   