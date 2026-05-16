from __future__ import annotations


class BoardUI:
    def __init__(self, event_queue=None, state_queue=None, square_px: int = 64) -> None:
        self.event_queue = event_queue
        self.state_queue = state_queue
        self.square_px = square_px

    def run(self) -> None:
        try:
            import pygame
        except Exception as exc:
            raise RuntimeError("pygame is required to run the board UI") from exc
        pygame.init()
        screen = pygame.display.set_mode((self.square_px * 8 + 220, self.square_px * 8))
        pygame.display.set_caption("MuJoCo Chess")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            screen.fill((32, 32, 32))
            for file_idx in range(8):
                for rank_idx in range(8):
                    color = (220, 205, 175) if (file_idx + rank_idx) % 2 == 0 else (82, 115, 92)
                    pygame.draw.rect(screen, color, (file_idx * self.square_px, (7 - rank_idx) * self.square_px, self.square_px, self.square_px))
            pygame.display.flip()
        pygame.quit()
