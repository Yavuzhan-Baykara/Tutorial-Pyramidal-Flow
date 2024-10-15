import os
import argparse
import requests
from tqdm import tqdm
from datasets import load_dataset
import datasets
import pandas as pd

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

    # Videoların kaydedileceği listeler
    video_records = []

    # Videoları indir
    for i, video_info in enumerate(tqdm(dataset)):
        video_url = video_info['contentUrl']
        video_id = video_info['videoid']
        save_path = os.path.join(args.output_dir, f"{video_id}.mp4")
        
        # Video indirme işlemi
        download_video(video_url, save_path)

        # Her video için kayıt bilgisi oluştur
        video_records.append({
            'video_id': video_id,
            'video_url': video_url,
            'saved_path': save_path
        })

        # Her 100 videoda bir çıktı yazdır
        if i % 100 == 0:
            print(f"{i} video indirildi...")

    # İndirilen videoları CSV dosyasına kaydetme
    df = pd.DataFrame(video_records)
    df.to_csv(args.output_csv, index=False)

    print(f"Veri seti CSV dosyası olarak kaydedildi: {args.output_csv}")

# Argümanları ayarlama
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face WebVid-10M Video Downloader")
    
    # Komut satırından girilecek argümanlar
    parser.add_argument('--dataset_name', type=str, default="TempoFunk/webvid-10M", help="Hugging Face veri seti adı")
    parser.add_argument('--output_dir', type=str, default="./webvid_videos", help="Videoların indirileceği klasör")
    parser.add_argument('--max_videos', type=int, default=100, help="İndirilecek maksimum video sayısı")
    parser.add_argument('--output_csv', type=str, default="/kaggle/working/webvid_dataset.csv", help="Videoların kaydedileceği CSV dosyası")

    args = parser.parse_args()
    
    # Ana fonksiyonu çalıştır
    main(args)