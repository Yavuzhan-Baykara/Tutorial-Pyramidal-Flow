import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import argparse

class WebVidDataset(Dataset):
    def __init__(self, data_dir: str, metadata_file: str, max_videos: int = 100, transform=None, video_max_len: int = 16):
        """
        WebVid-10M veri seti için veri yükleyici.
        
        Args:
            data_dir (str): Video dosyalarının bulunduğu dizin.
            metadata_file (str): CSV dosyasının yolu.
            max_videos (int): Yüklenecek maksimum video sayısı (örneğin ilk 100 video).
            transform (callable, optional): Videolara uygulanacak ön işleme adımları.
            video_max_len (int): Videonun maksimum kare sayısı (varsayılan: 16).
        """
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.transform = transform
        self.video_max_len = video_max_len

        # CSV dosyasından belirlenen sayıda videoyu yükle
        self.metadata = pd.read_csv(self.metadata_file).head(max_videos)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Belirtilen indeksteki video ve açıklamayı döndürür.
        
        Args:
            idx (int): Veriden bir örnek almak için indeks.
            
        Returns:
            video_tensor (torch.Tensor): Video verisi.
            caption (str): Videoya ait açıklama.
        """
        video_info = self.metadata.iloc[idx]
        video_path = os.path.join(self.data_dir, video_info['video_path'])
        caption = video_info['caption']

        # Videoyu yükleme
        video_frames = self.load_video(video_path)

        # Eğer transform fonksiyonu tanımlandıysa videoya uygula
        if self.transform:
            video_frames = self.transform(video_frames)

        # Video tensor'a dönüştür
        video_tensor = torch.stack([torch.tensor(frame) for frame in video_frames])

        return video_tensor, caption

    def load_video(self, video_path: str):
        """
        OpenCV kullanarak videoyu yükler ve karelere ayırır.

        Args:
            video_path (str): Yüklenecek video dosyasının yolu.
        
        Returns:
            frames (list): Video kareleri.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= self.video_max_len:
                break
            frame = cv2.resize(frame, (128, 128))  # Videoları 128x128 boyutuna ölçeklendir
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV RGB formatına çevir
            frames.append(frame)
            frame_count += 1

        cap.release()
        return frames

def main(args):
    # Veri yükleyiciyi başlat
    dataset = WebVidDataset(data_dir=args.data_dir, metadata_file=args.metadata_file, max_videos=args.max_videos)
    
    # Veri yükleyiciyi DataLoader ile bağlayın
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # İlk batch'i inceleyin
    for video_tensor, caption in dataloader:
        print(f"Video shape: {video_tensor.shape}, Caption: {caption}")
        break

if __name__ == "__main__":
    # Argümanları tanımla
    parser = argparse.ArgumentParser(description="WebVid-10M dataset loader")
    parser.add_argument('--data_dir', type=str, required=True, help='Video dosyalarının bulunduğu dizin')
    parser.add_argument('--metadata_file', type=str, required=True, help='CSV metadata dosyasının yolu')
    parser.add_argument('--max_videos', type=int, default=100, help='Yüklenecek maksimum video sayısı')
    
    # Argümanları ayrıştır
    args = parser.parse_args()

    # Ana fonksiyonu çalıştır
    main(args)