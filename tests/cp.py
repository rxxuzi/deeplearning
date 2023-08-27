import time

class Progress:
    def __init__(self, max, progress = 0,percentage = 10):
        self.max = max
        self.percentage = percentage
        self.progress = progress

    def update(self, epoch):
        self.progress = (epoch) / self.max
        self.percentage = self.progress * 100

    def progress_bar(self):
        if self.progress < 0 or self.progress > 1:
            print("p should be between 0 and 1")
            return

        total_width = 23  # Total width of the bar
        filled_width = int(self.progress * total_width)
        empty_width = total_width - filled_width

        if self.progress == 1:
            bar = "[" + "#" * filled_width + "=" * empty_width + "]" + "100.00%"
        else:
            bar = "[" + "#" * filled_width + "=" * empty_width + "]" + f"{self.percentage:.2f}%"

        print(bar)

    def update_bar(self, epoch):
        self.progress = (epoch) / self.max
        self.percentage = self.progress * 100
        self.progress_bar()


max = 100
sleep_time = 0.03
progress = Progress(max)
for i in range(max):
    time.sleep(sleep_time)
    if i % 2 == 0:
        progress.update(i)
        progress.progress_bar()

progress.update(max)
progress.progress_bar()
print("Done")