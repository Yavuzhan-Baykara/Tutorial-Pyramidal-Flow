import os
import argparse
import requests
from tqdm import tqdm
from datasets import load_dataset
import datasets

# Hugging Face veri setinden videoları indirip kaydeden bir fonksiyon
def download_video(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# Ana fonksiyon
def main(args):
    # Hugging Face veri seti yükleme
    datasets.logging.set_verbosity_warning()
    dataset = load_dataset(args.dataset_name, split=f"train[:{args.max_videos}]")  # Belirtilen video sayısı kadar indir

    # Çıkış klasörünü oluştur
    os.makedirs(args.output_dir, exist_ok=True)

    # Toplam video sayısını yazdır
    print(f"İndirilecek toplam video sayısı: {len(dataset)}")

    # Videoları indir
    for i, video_info in enumerate(tqdm(dataset)):
        video_url = video_info['contentUrl']
        video_id = video_info['videoid']
        save_path = os.path.join(args.output_dir, f"{video_id}.mp4")
        
        download_video(video_url, save_path)

        # Her 100 videoda bir çıktı yazdır
        if i % 100 == 0:
            print(f"{i} video indirildi...")

# Argümanları ayarlama
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face WebVid-10M Video Downloader")
    
    # Komut satırından girilecek argümanlar
    parser.add_argument('--dataset_name', type=str, default="TempoFunk/webvid-10M", help="Hugging Face veri seti adı")
    parser.add_argument('--output_dir', type=str, default="./webvid_videos", help="Videoların indirileceği klasör")
    parser.add_argument('--max_videos', type=int, default=100, help="İndirilecek maksimum video sayısı")
    
    args = parser.parse_args()
    
    # Ana fonksiyonu çalıştır
    main(args)