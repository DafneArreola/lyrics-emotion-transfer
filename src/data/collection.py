"""
Genius API lyrics collection script.
Scrapes lyrics for curated artist lists with rate limiting and error handling.
"""

import lyricsgenius
import yaml
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeniusCollector:
    """Collect lyrics from Genius API for specified artists."""
    
    def __init__(self, api_token: str, config_path: str = "configs/artists.yaml"):
        """
        Initialize collector.
        
        Args:
            api_token: Genius API access token
            config_path: Path to artist configuration YAML
        """
        self.genius = lyricsgenius.Genius(
            api_token,
            sleep_time=0.5,  # Rate limiting: 2 requests/second
            timeout=15,
            retries=3,
            remove_section_headers=True,  # Remove [Verse], [Chorus] tags
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)", "(Acoustic)", "(Demo)"]
        )
        
        # Load artist lists
        with open(config_path, 'r') as f:
            self.artists_by_genre = yaml.safe_load(f)
        
        self.raw_dir = Path("data/raw/genius_pulls")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_artist_songs(
        self, 
        artist_name: str, 
        max_songs: int = 100,
        genre: str = None
    ) -> Optional[Dict]:
        """
        Collect all songs for a single artist.
        
        Args:
            artist_name: Artist name to search
            max_songs: Maximum songs to retrieve per artist
            genre: Genre label for metadata
            
        Returns:
            Dictionary with artist metadata and songs, or None if failed
        """
        try:
            logger.info(f"Fetching songs for {artist_name} ({genre})...")
            
            # Search for artist
            artist = self.genius.search_artist(
                artist_name, 
                max_songs=max_songs,
                sort="popularity"
            )
            
            if artist is None:
                logger.warning(f"Artist not found: {artist_name}")
                return None
            
            # Extract song data
            songs_data = []
            for song in artist.songs:
                songs_data.append({
                    'title': song.title,
                    'lyrics': song.lyrics,
                    'url': song.url,
                    'pageviews': song.stats.pageviews if song.stats else None,
                    'release_date': song.release_date_for_display,
                    'featured_artists': [a.name for a in song.featured_artists] if song.featured_artists else []
                })
            
            artist_data = {
                'artist_name': artist.name,
                'artist_id': artist.id,
                'genre': genre,
                'num_songs': len(songs_data),
                'songs': songs_data,
                'collection_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Collected {len(songs_data)} songs for {artist_name}")
            return artist_data
            
        except Exception as e:
            logger.error(f"Error collecting {artist_name}: {str(e)}")
            return None
    
    def save_artist_data(self, artist_data: Dict, genre: str):
        """Save individual artist data to JSON."""
        if artist_data is None:
            return
        
        filename = f"{genre}_{artist_data['artist_name'].replace(' ', '_')}.json"
        filepath = self.raw_dir / genre / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(artist_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved to {filepath}")
    
    def collect_genre(self, genre: str, max_songs_per_artist: int = 100):
        """
        Collect all songs for a genre.
        
        Args:
            genre: Genre key from config (pop, hiphop, country, rock)
            max_songs_per_artist: Max songs per artist
        """
        artists = self.artists_by_genre.get(genre, [])
        logger.info(f"Starting collection for {genre}: {len(artists)} artists")
        
        genre_stats = {
            'total_artists': len(artists),
            'successful': 0,
            'failed': 0,
            'total_songs': 0
        }
        
        for i, artist_name in enumerate(artists, 1):
            logger.info(f"[{i}/{len(artists)}] Processing {artist_name}...")
            
            artist_data = self.collect_artist_songs(
                artist_name, 
                max_songs=max_songs_per_artist,
                genre=genre
            )
            
            if artist_data:
                self.save_artist_data(artist_data, genre)
                genre_stats['successful'] += 1
                genre_stats['total_songs'] += artist_data['num_songs']
            else:
                genre_stats['failed'] += 1
            
            # Rate limiting buffer
            time.sleep(1)
        
        logger.info(f"\n{genre.upper()} Collection Complete:")
        logger.info(f"  Successful: {genre_stats['successful']}/{genre_stats['total_artists']}")
        logger.info(f"  Total songs: {genre_stats['total_songs']}")
        logger.info(f"  Failed: {genre_stats['failed']}")
        
        return genre_stats
    
    def collect_all_genres(self, max_songs_per_artist: int = 100):
        """Collect lyrics for all genres."""
        all_stats = {}
        
        for genre in ['pop', 'hiphop', 'country', 'rock']:
            logger.info(f"\n{'='*60}\nCollecting {genre.upper()}\n{'='*60}")
            stats = self.collect_genre(genre, max_songs_per_artist)
            all_stats[genre] = stats
            
            # Save intermediate progress
            with open(self.raw_dir / 'collection_stats.json', 'w') as f:
                json.dump(all_stats, f, indent=2)
        
        return all_stats


def main():
    """Main collection pipeline."""
    # Load API token from environment variable
    api_token = os.getenv('GENIUS_API_TOKEN')
    
    if not api_token:
        logger.error("GENIUS_API_TOKEN not set. Please set environment variable.")
        logger.info("Get token from: https://genius.com/api-clients")
        return
    
    collector = GeniusCollector(api_token)
    
    # Collect all genres (100 songs per artist -> ~2000 songs per genre)
    stats = collector.collect_all_genres(max_songs_per_artist=100)
    
    logger.info("\n" + "="*60)
    logger.info("COLLECTION COMPLETE")
    logger.info("="*60)
    for genre, data in stats.items():
        logger.info(f"{genre.upper()}: {data['total_songs']} songs from {data['successful']} artists")


if __name__ == "__main__":
    main()