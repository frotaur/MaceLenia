import pygame, os, json

from modules.main_utils import Camera
from modules.models import *
from modules.main_utils import launch_video, add_frame, print_screen
from modules.main_utils import TextBlock, DropdownMenu, InputField, render_text_blocks
from pathlib import Path

import pygame_chart as pyc

from modules.models.diffusion_lenia_cross_channel import DiffusionLeniaCrossChannel
from modules.models.evolvable_diffusion_lenia import EvolvableDiffusionLenia
from modules.models.flow_lenia import FlowLenia

cur_dir = Path(__file__).parent
font_path = (cur_dir / "modules" / "main_utils" / "interface" / "AldotheApache.ttf").as_posix()
std_help = (cur_dir / "modules" / "main_utils" / "interface" / "std_help.json").as_posix()

with open(std_help, "r") as f:
    std_help = json.load(f)

pygame.init()


def gameloop(screen: tuple[int], world: tuple[int], device: str):
    # Define available automaton classes
    automaton_options = {
        "DiffusionLenia": lambda h, w: DiffusionLenia(
            (1, h, w),
            dt=0.1,
            num_channels=3,
            device=device,
            has_food=True,
            save_dir="saved_diff_lenia",
            interest_files=(cur_dir / "demo_params").as_posix(),
        ),
        "DiffusionLeniaCrossChannel": lambda h, w: DiffusionLeniaCrossChannel(
            (1, h, w),
            dt=0.1,
            num_channels=3,
            device=device,
            has_food=False,
            save_dir="saved_diff_lenia",
            interest_files=(cur_dir / "demo_params").as_posix(),
        ),
        "Lenia": lambda h, w: MCLenia(
            (1, h, w),
            dt=0.1,
            num_channels=3,
            save_dir="saved_lenia",
            interest_files=(cur_dir / "demo_params").as_posix(),
            device=device,

        ),
        "FlowLenia": lambda h, w: FlowLenia(
            (1, h, w),
            dt=0.1,
            num_channels=3,
            save_dir="saved_lenia",
            interest_files=(cur_dir / "demo_params").as_posix(),
            device=device,
            has_food=True,
            ),

        "EvolvableDiffusionLenia": lambda h, w: EvolvableDiffusionLenia(
            (16, h, w),
            dt=0.1,
            num_channels=3,
            save_dir="saved_diff_lenia",
            interest_files=(cur_dir / "demo_params").as_posix(),
            device=device,
            has_food=False,
        ),
    }
    sW, sH = screen

    Mr = 5
    Mm = 5

    # Automaton world size
    W, H = world
    device = device

    fps = 60  # Visualization (target) frames per second
    video_fps = 60  # Video frames per second

    text_size = int(sH / 45)
    title_size = int(text_size * 1.3)
    font = pygame.font.Font(font_path, size=text_size)
    font_title = pygame.font.Font(font_path, size=title_size)

    screen = pygame.display.set_mode((sW, sH), flags=pygame.RESIZABLE)
    figure = pyc.Figure(screen, 0.8 * sW, sH // 2, sW * 0.2, sH * 0.2)




    clock = pygame.time.Clock()
    running = True
    camera = Camera(W, H)
    camera.resize(sW, sH)
    zoom = min(sW / W, sH / H)
    camera.zoom = zoom

    # Booleans for the main loop
    stopped = True
    recording = False
    launch_vid = True
    display_help = True
    writer = None

    # Then when initializing the first automaton:
    initial_automaton = "DiffusionLenia"
    auto = automaton_options[initial_automaton](H, W)

    description, help_text = auto.get_help()


    def display_fig(figure, data):

        x = [i for i in range(len(data))]
        figure.line('Chart1', x, data)
        figure.draw()

    def make_text_blocks(description, help_text, std_help, font, font_title):
        text_blocks = [
            TextBlock(description, "up_sx", (74, 101, 176), font_title),
            TextBlock("\n", "up_sx", (230, 230, 230), font),
        ]
        for section in std_help["sections"]:
            text_blocks.append(TextBlock(section["title"], "up_sx", (230, 89, 89), font))
            for command, description in section["commands"].items():
                text_blocks.append(TextBlock(f"{command} -> {description}", "up_sx", (230, 230, 230), font))
            text_blocks.append(TextBlock("\n", "up_sx", (230, 230, 230), font))
        text_blocks.append(TextBlock("Automaton controls", "below_sx", (230, 89, 89), font))
        text_blocks.append(TextBlock(help_text, "below_sx", (230, 230, 230), font))
        return text_blocks

    def display_live_text(auto: Automaton, font, screen):
        sW, sH = screen.get_size()
        stringu = auto.get_string_state()
        text = font.render(stringu, True, (255, 255, 255))
        inf_size = 10
        text_rect = text.get_rect()
        text_rect.centerx = sW // 2
        text_rect.bottom = sH - inf_size

        bg_rect = text_rect.inflate(inf_size, inf_size)
        # Create a semi-transparent surface for the background
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 128))

        # Draw the background and text
        screen.blit(bg_surface, bg_rect)
        screen.blit(text, text_rect)

    text_blocks = make_text_blocks(description, help_text, std_help, font, font_title)

    # Update these initial sizes to be relative to screen size
    button_width = int(sW * 0.15)  # 15% of screen width
    button_height = int(sH * 0.05)  # 5% of screen height
    input_width = int(sW * 0.05)  # 5% of screen width
    input_height = int(sH * 0.05)  # 5% of screen height
    margin = int(sH * 0.02)  # 2% of screen height

    dropdown = DropdownMenu(
        screen=screen,
        width=button_width,
        height=button_height,
        font=font,
        options=automaton_options,
        default_option=initial_automaton,
        margin=margin,
    )

    w_input = InputField(
        screen=screen,
        width=input_width,
        height=input_height,
        font=font,
        label="Width",
        initial_value=W,
        margin=margin,
        index=0,
    )

    h_input = InputField(
        screen=screen,
        width=input_width,
        height=input_height,
        font=font,
        label="Height",
        initial_value=H,
        margin=margin,
        index=1,
    )

    fps_input = InputField(
        screen=screen,
        width=input_width,
        height=input_height,
        font=font,
        label="FPS",
        initial_value=fps,
        margin=margin,
        index=2,
    )

    mr_input = InputField(
        screen=screen,
        width=input_width,
        height=input_height,
        font=font,
        label="Mutation Rate",
        initial_value=Mr,
        margin=margin,
        index=3,
    )
    mm_input = InputField(
        screen=screen,
        width=input_width,
        height=input_height,
        font=font,
        label="Mutation Magnitude",
        initial_value=Mm,
        margin=margin,
        index=4,
    )

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            camera.handle_event(event)  # Handle the camera events

            if event.type == pygame.KEYDOWN:  # Universal key interactivity
                if event.key == pygame.K_SPACE:
                    stopped = not (stopped)
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    recording = not recording
                    if not launch_vid and writer is not None:
                        launch_vid = True
                        writer.release()
                if event.key == pygame.K_p:
                    print_screen(auto.worldsurface)
                if event.key == pygame.K_s:
                    auto.step()
                if event.key == pygame.K_h:
                    display_help = not display_help
                if event.key == pygame.K_c:
                    current_sW, current_sH = screen.get_size()
                    camera = Camera(W, H)
                    camera.resize(current_sW, current_sH)
                    zoom = min(current_sW / W, current_sH / H)
                    camera.zoom = zoom

            if event.type == pygame.VIDEORESIZE:
                # Get current window size and new window size
                old_w, old_h = screen.get_size()
                new_w, new_h = event.w, event.h

                # Calculate scale factors
                scale_w = new_w / old_w
                scale_h = new_h / old_h

                # Update camera with new screen dimensions and scale position and zoom
                camera.resize(new_w, new_h)
                camera.position.x *= scale_w
                camera.position.y *= scale_h
                camera.zoom *= min(scale_w, scale_h)  # Use minimum scale to preserve aspect ratio
                camera.updateFov()

                # Calculate new sizes based on new dimensions
                button_width = int(new_w * 0.15)
                button_height = int(new_h * 0.05)
                input_width = int(new_w * 0.05)
                input_height = int(new_h * 0.05)
                margin = int(new_h * 0.02)

                #figure.width = int(0.2 * new_w)
                #figure.height = int(0.2 * new_h)
                figure.x = int(0.8*new_w)
                figure.y = int(0.5*new_h)


                # Update text sizes
                text_size = int(new_h / 45)
                title_size = int(text_size * 1.5)
                font = pygame.font.Font(font_path, size=text_size)
                font_title = pygame.font.Font(font_path, size=title_size)

                # Update UI elements with new sizes and font
                dropdown.resize(button_width, button_height, margin, font)
                w_input.resize(input_width, input_height, margin, font)
                h_input.resize(input_width, input_height, margin, font)
                fps_input.resize(input_width, input_height, margin, font)
                mr_input.resize(input_width, input_height, margin, font)
                mm_input.resize(input_width, input_height, margin, font)

                # Update text blocks with new font
                text_blocks = make_text_blocks(description, help_text, std_help, font, font_title)

            auto.process_event(event, camera)  # Process the event in the automaton

            if dropdown.handle_event(event):  # Handle dropdown event
                auto = automaton_options[dropdown.current_option](H, W)
                # Update help text
                description, help_text = auto.get_help()
                text_blocks = make_text_blocks(description, help_text, std_help, font, font_title)

            if display_help:  # Handle input field events
                if w_input.handle_event(event):
                    new_w = w_input.get_value()
                    if new_w and new_w > 0:
                        W = new_w
                        current_sW, current_sH = screen.get_size()
                        # Recreate automaton with new size
                        auto = automaton_options[dropdown.current_option](H, W)
                        camera = Camera(W, H)
                        camera.resize(current_sW, current_sH)
                        zoom = min(current_sW / W, current_sH / H)
                        camera.zoom = zoom

                if h_input.handle_event(event):
                    new_h = h_input.get_value()
                    if new_h and new_h > 0:
                        H = new_h
                        current_sW, current_sH = screen.get_size()
                        # Recreate automaton with new size
                        auto = automaton_options[dropdown.current_option](H, W)
                        camera = Camera(W, H)
                        camera.resize(current_sW, current_sH)
                        zoom = min(current_sW / W, current_sH / H)
                        camera.zoom = zoom

                if fps_input.handle_event(event):
                    new_fps = fps_input.get_value()
                    if new_fps and new_fps > 0:
                        fps = new_fps

                if mm_input.handle_event(event):
                    new_mm = mm_input.get_value()
                    if new_mm and new_mm > 0:
                        Mm = new_mm
                        auto.mm = Mm/100

                if mr_input.handle_event(event):
                    new_mr = mr_input.get_value()
                    if new_mr and new_mr > 0:
                        Mr = new_mr
                        auto.mr = Mr/100



        if not stopped:
            auto.step()  # step the automaton

        auto.draw()  # draw the worldstate
        world_surface = auto.worldsurface

        # Clear the screen
        screen.fill((0, 0, 0))



        # Draw the scaled surface on the window
        zoomed_surface = camera.apply(world_surface, border=False)
        screen.blit(zoomed_surface, (0, 0))

        if recording:
            if launch_vid:  # If the video is not launched, we create it
                launch_vid = False
                writer = launch_video((H, W), video_fps, "mp4v")
            add_frame(
                writer, world_surface
            )  # (in the future, we may add the zoomed frame instead of the full frame)
            pygame.draw.circle(screen, (255, 0, 0), (sW - 10, 15), 7)

        if display_help:
            render_text_blocks(
                screen, [TextBlock(f"FPS: {int(clock.get_fps())}", "up_dx", (255, 89, 89), font)]
            )
            render_text_blocks(screen, text_blocks)

        # Draw dropdown (before pygame.display.flip())
        dropdown.draw(screen, display_text=display_help)

        # Draw input fields
        if display_help:
            w_input.draw()
            h_input.draw()
            fps_input.draw()
            if hasattr(auto, "masses"):
                mr_input.draw()
                mm_input.draw()

        display_live_text(auto, font, screen)
        # Update the screen
        if hasattr(auto, "masses"):

            data = auto.masses
            figure.set_ylim((min(data)-5, max(data)+6))
            display_fig(figure,data)
        pygame.display.flip()

        clock.tick(fps)  # limits FPS to 60

    pygame.quit()


if __name__ == "__main__":

    gameloop((1920, 1080), (500, 500), "cuda:0")
