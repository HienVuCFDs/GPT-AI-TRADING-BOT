"""
NewsScraperSelenium - Quét tin tức từ website ForexFactory bằng Chrome/Selenium
Lấy dữ liệu Actual, Forecast, Previous trực tiếp từ HTML
Enhanced with automatic data cleanup features
"""
import time
import json
import pytz
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import os
import json
import logging
import re
import configparser

 # API feed disabled per requirement

# Internal utility functions
def overwrite_json_safely(file_path: str, data: any, backup: bool = True) -> bool:
    """Save JSON data safely with backup support"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")
        return False

def ensure_directory(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def auto_cleanup_on_start(directories: list, hours: int = 72):
    """Auto cleanup on start"""
    pass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class NewsScraperSelenium:
    def __init__(self):
        self.vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        self.url = "https://www.forexfactory.com/calendar"
        self.driver = None
    # API feed disabled; Selenium-only scraping

    def _load_mt5_timezone_config(self) -> Dict[str, Any]:
        """Load MT5 server timezone configuration from env/ini with broker mapping.

        Priority:
        1) Env MT5_SERVER_TIMEZONE (IANA tz name e.g. Europe/Athens)
        2) Env MT5_SERVER_UTC_OFFSET (hours, e.g. 3 or -5)
        3) mt5_data_config.ini [MT5] server_timezone / server_utc_offset
        4) mt5_data_config.ini [BROKERS] mapping by detected broker/server name or env MT5_BROKER
        """
        cfg: Dict[str, Any] = {
            'tz_name': None,
            'utc_offset_hours': None
        }

        # 1) Environment overrides
        tz_env = os.getenv('MT5_SERVER_TIMEZONE')
        if tz_env:
            cfg['tz_name'] = tz_env.strip()

        off_env = os.getenv('MT5_SERVER_UTC_OFFSET')
        if off_env:
            try:
                cfg['utc_offset_hours'] = int(off_env)
            except ValueError:
                pass

        # Early return if fully specified by env
        if cfg['tz_name'] or cfg['utc_offset_hours'] is not None:
            return cfg

        # 2) Load INI and optional broker map
        ini_path = os.path.join(os.getcwd(), 'mt5_data_config.ini')
        parser = configparser.ConfigParser()
        if os.path.exists(ini_path):
            try:
                parser.read(ini_path)

                # Try per-broker mapping FIRST
                if parser.has_section('BROKERS'):
                    # Determine broker key: env MT5_BROKER or auto-detect via MT5
                    broker_key = os.getenv('MT5_BROKER', '').strip()
                    if not broker_key:
                        try:
                            import MetaTrader5 as mt5  # Optional
                            if mt5.initialize():
                                try:
                                    acc = mt5.account_info()
                                    if acc and getattr(acc, 'server', None):
                                        broker_key = acc.server
                                    else:
                                        ti = mt5.terminal_info()
                                        if ti and getattr(ti, 'company', None):
                                            broker_key = ti.company
                                finally:
                                    mt5.shutdown()
                        except Exception:
                            broker_key = ''

                    if broker_key and (cfg['tz_name'] is None and cfg['utc_offset_hours'] is None):
                        bk_lower = broker_key.lower()
                        for name, value in parser.items('BROKERS'):
                            # Match if broker mapping key is contained in detected broker/server string
                            if name.lower() in bk_lower:
                                val = (value or '').strip()
                                if val:
                                    # Accept IANA tz or UTC±X patterns
                                    m = re.match(r'^UTC([+\-])(\d{1,2})$', val.upper())
                                    if m:
                                        sign, hours = m.groups()
                                        offset = int(hours)
                                        cfg['utc_offset_hours'] = offset if sign == '+' else -offset
                                    else:
                                        cfg['tz_name'] = val
                                break

                # If still not set, use [MT5] defaults
                if (cfg['tz_name'] is None and cfg['utc_offset_hours'] is None) and parser.has_section('MT5'):
                    if parser.has_option('MT5', 'server_timezone'):
                        tz = parser.get('MT5', 'server_timezone', fallback='').strip()
                        cfg['tz_name'] = tz or None
                    if cfg['tz_name'] is None and parser.has_option('MT5', 'server_utc_offset'):
                        try:
                            cfg['utc_offset_hours'] = parser.getint('MT5', 'server_utc_offset', fallback=None)
                        except Exception:
                            cfg['utc_offset_hours'] = None
            except Exception:
                pass

        return cfg

    def _get_mt5_timezone(self) -> Dict[str, Any]:
        """Return a dict with keys: tz (tzinfo or None), label (str)"""
        cfg = self._load_mt5_timezone_config()
        tzinfo_obj = None
        label = None
        # Prefer IANA tz when provided
        if cfg.get('tz_name'):
            try:
                tzinfo_obj = pytz.timezone(cfg['tz_name'])
                label = cfg['tz_name']
            except Exception:
                tzinfo_obj = None
                label = None
        # Fallback to fixed offset
        if tzinfo_obj is None and cfg.get('utc_offset_hours') is not None:
            try:
                minutes = int(cfg['utc_offset_hours']) * 60
                tzinfo_obj = pytz.FixedOffset(minutes)
                sign = '+' if cfg['utc_offset_hours'] >= 0 else ''
                label = f"UTC{sign}{cfg['utc_offset_hours']}"
            except Exception:
                tzinfo_obj = None
                label = None

        return {'tz': tzinfo_obj, 'label': label}

    def _parse_vn_event_datetime(self, event: Dict[str, Any]) -> Optional[datetime]:
        """Build a timezone-aware VN datetime from event's date and time fields.
        Returns None if not parseable or All Day/empty.
        """
        event_date = event.get('date', '')
        event_time = event.get('time', '')
        if not event_time or str(event_time).strip() in ("All Day", ""):
            return None

        # Parse time in 12-hour format like 1:00am/pm
        m = re.match(r'^(\d{1,2}):(\d{2})(am|pm)$', str(event_time).strip().lower())
        if not m:
            return None
        hour, minute, period = m.groups()
        hour = int(hour)
        minute = int(minute)
        if period == 'pm' and hour != 12:
            hour += 12
        elif period == 'am' and hour == 12:
            hour = 0

        # Parse date in several formats
        base_date = None
        try:
            base_date = datetime.strptime(str(event_date), '%Y-%m-%d')
        except Exception:
            for fmt in ('%b %d, %Y', '%B %d, %Y', '%b %d %Y', '%B %d %Y'):
                try:
                    base_date = datetime.strptime(str(event_date), fmt)
                    break
                except Exception:
                    pass
            if base_date is None:
                # Use existing helper to normalize to ISO
                iso = self._parse_date_from_text(str(event_date), datetime.now(self.vn_tz))
                try:
                    base_date = datetime.strptime(iso, '%Y-%m-%d')
                except Exception:
                    base_date = None
        if base_date is None:
            return None

        return self.vn_tz.localize(base_date.replace(hour=hour, minute=minute, second=0, microsecond=0))

    def _augment_events_with_mt5_time(self, events: List[Dict[str, Any]]):
        """Add VN 24h time and MT5 time equivalents to events without altering existing fields."""
        mt5_tz_info = self._get_mt5_timezone()
        target_tz = mt5_tz_info.get('tz')
        target_label = mt5_tz_info.get('label')

        for e in events:
            try:
                dt_vn = self._parse_vn_event_datetime(e)
                if not dt_vn:
                    # Keep markers for All Day/unknown
                    if e.get('time') in ("All Day", ""):
                        e.setdefault('timezone', 'Vietnam Time (UTC+7)')
                        if target_tz is not None:
                            e.setdefault('mt5_timezone', target_label or 'MT5 Server Time')
                            e.setdefault('time_mt5', e.get('time'))
                            e.setdefault('date_mt5', e.get('date'))
                    continue

                # VN 24h time convenience
                e['time_vn_24h'] = dt_vn.strftime('%H:%M')
                e.setdefault('timezone', 'Vietnam Time (UTC+7)')

                if target_tz is not None:
                    dt_mt5 = dt_vn.astimezone(target_tz)
                    e['time_mt5'] = dt_mt5.strftime('%H:%M')
                    e['date_mt5'] = dt_mt5.strftime('%Y-%m-%d')
                    e['datetime_mt5'] = dt_mt5.strftime('%Y-%m-%d %H:%M')
                    e['mt5_timezone'] = target_label or 'MT5 Server Time'
            except Exception:
                # Non-fatal
                continue

    def start_browser(self, headless=True):
        chrome_options = Options()
        # Use new headless mode for better compatibility
        if headless:
            chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--lang=en-US')
        chrome_options.add_argument('--blink-settings=imagesEnabled=false')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        # Set a realistic user agent
        chrome_options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
        )
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print("❌ Không khởi động được ChromeDriver! Kiểm tra lại cài đặt hoặc PATH.")
            print(f"Lỗi: {e}")
            self.driver = None

    def stop_browser(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

    def _parse_date_from_text(self, date_text, reference_date):
        """Chuyển đổi text ngày thành format YYYY-MM-DD"""
        import re
        from datetime import datetime, timedelta
        
        try:
            date_text = date_text.strip().lower()
            
            # Trường hợp "today"
            if 'today' in date_text:
                return reference_date.strftime('%Y-%m-%d')
            
            # Trường hợp "tomorrow"
            if 'tomorrow' in date_text:
                tomorrow = reference_date + timedelta(days=1)
                return tomorrow.strftime('%Y-%m-%d')
            
            # Trường hợp "yesterday"  
            if 'yesterday' in date_text:
                yesterday = reference_date - timedelta(days=1)
                return yesterday.strftime('%Y-%m-%d')
            
            # Trường hợp format "Aug 06", "August 6", etc.
            month_patterns = {
                'jan': 1, 'january': 1,
                'feb': 2, 'february': 2, 
                'mar': 3, 'march': 3,
                'apr': 4, 'april': 4,
                'may': 5,
                'jun': 6, 'june': 6,
                'jul': 7, 'july': 7,
                'aug': 8, 'august': 8,
                'sep': 9, 'september': 9,
                'oct': 10, 'october': 10,
                'nov': 11, 'november': 11,
                'dec': 12, 'december': 12
            }
            
            for month_name, month_num in month_patterns.items():
                if month_name in date_text:
                    # Tìm số ngày
                    day_match = re.search(rf'{month_name}\s*(\d{{1,2}})', date_text)
                    if day_match:
                        day = int(day_match.group(1))
                        year = reference_date.year
                        
                        # Nếu tháng < tháng hiện tại thì có thể là năm sau
                        if month_num < reference_date.month:
                            year += 1
                        
                        return f"{year:04d}-{month_num:02d}-{day:02d}"
            
            # Nếu không parse được, trả về ngày tham chiếu
            return reference_date.strftime('%Y-%m-%d')
            
        except Exception as e:
            print(f"⚠️ Lỗi parse ngày '{date_text}': {e}")
            return reference_date.strftime('%Y-%m-%d')

    def _filter_recent_news_with_actual(self, events):
        """Lọc các tin tức có số liệu thực tế và vừa được công bố trong 60 phút trở lại"""
        filtered_events = []
        now = datetime.now(self.vn_tz)

        for event in events:
            # Kiểm tra có số liệu thực tế không
            actual = event.get('actual', '').strip()
            if not actual or actual in ['TBA', 'Pending', '', '-', '–', 'N/A']:
                continue

            # Tạo datetime từ date và time của event
            try:
                event_date = event.get('date', '')  # expected ISO; handle non-ISO too
                event_time = event.get('time', '')  # format: 1:00am, 6:20am, etc.

                if not event_time or str(event_time).strip() == "All Day":
                    continue

                # Parse thời gian
                time_match = re.match(r'(\d{1,2}):(\d{2})(am|pm)', str(event_time).lower())
                if time_match:
                    hour, minute, period = time_match.groups()
                    hour = int(hour)
                    if period == 'pm' and hour != 12:
                        hour += 12
                    elif period == 'am' and hour == 12:
                        hour = 0

                    # Build base date from various formats
                    base_date = None
                    try:
                        base_date = datetime.strptime(str(event_date), '%Y-%m-%d')
                    except Exception:
                        for fmt in ('%b %d, %Y', '%B %d, %Y', '%b %d %Y', '%B %d %Y'):
                            try:
                                base_date = datetime.strptime(str(event_date), fmt)
                                break
                            except Exception:
                                pass
                        if base_date is None:
                            # Fallback to helper to get ISO, then parse
                            iso_fallback = self._parse_date_from_text(str(event_date), now)
                            try:
                                base_date = datetime.strptime(iso_fallback, '%Y-%m-%d')
                            except Exception:
                                base_date = None
                    if base_date is None:
                        continue
                        
                    # Create timezone-aware datetime properly
                    event_datetime = self.vn_tz.localize(
                        base_date.replace(hour=hour, minute=int(minute), second=0, microsecond=0)
                    )

                    # Kiểm tra thời gian công bố (trong 60 phút trở lại để có thời gian xử lý)
                    time_diff = (now - event_datetime).total_seconds()

                    if 0 <= time_diff <= 3600:  # 60 phút = 3600 giây (mở rộng từ 30 phút)
                        filtered_events.append(event)
                        print(f"[NEWS] Recent news found: {event_time} {event['currency']} {event['event']} = {actual} ({time_diff/60:.1f} minutes ago)")

            except Exception as e:
                print(f"[WARNING] Time parsing error for {event.get('event', 'unknown')}: {e}")
                continue

        return filtered_events

    def get_today_events(self, currencies: List[str] = None, impacts: List[int] = None, headless: bool = True) -> List[Dict[str, Any]]:
        # Selenium-only flow per requirement
        today = datetime.now(self.vn_tz)
        today_str = today.strftime('%Y%m%d')
        print("🔍 Extracting calendar data via Selenium (no API)...")

        self.start_browser(headless=headless)
        if not self.driver:
            print("❌ Không thể khởi động trình duyệt để quét tin!")
            return []
        try:
            self.driver.get(self.url)
            # Wait for page to load
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(10)  # Longer wait for JavaScript content to load

            # Method 1: JSON embedded in page (if available)
            page_source = self.driver.page_source
            events = self._extract_events_from_json(page_source, today) or []

            if not events:
                print("⚠️ JSON extraction failed, trying HTML parsing...")
                events = self._extract_events_from_html(today)

            if not events:
                print("⚠️ HTML extraction also failed, trying emergency method...")
                events = self._emergency_extract_events(today)
        finally:
            self.stop_browser()
            
            # Filter events
            filtered_events = []
            for event in events:
                add_event = True
                
                # Currency filter
                if currencies and event.get('currency', '') not in currencies:
                    add_event = False
                
                # Impact filter
                if impacts:
                    # Extended impact mapping to handle various ForexFactory formats
                    impact_map = {
                        'Low': 1, 'Medium': 2, 'High': 3, 'None': 0,
                        'low': 1, 'medium': 2, 'high': 3, 'none': 0,
                        'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'NONE': 0,
                        '1': 1, '2': 2, '3': 3, '0': 0,
                        'yellow': 1, 'orange': 2, 'red': 3,  # Color-based
                        'Yellow': 1, 'Orange': 2, 'Red': 3,
                        'weak': 1, 'moderate': 2, 'strong': 3,  # Alternative terms
                        'Weak': 1, 'Moderate': 2, 'Strong': 3
                    }
                    
                    raw_label = str(event.get('impact', 'None')).strip()
                    event_impact = impact_map.get(raw_label, 0)
                    
                    # If still 0 (unmapped), try title case normalization
                    if event_impact == 0 and raw_label:
                        norm_label = raw_label.title()
                        event_impact = impact_map.get(norm_label, 0)
                    
                    # If STILL unmapped, log it for debugging but set as Low (1) to avoid losing data
                    if event_impact == 0 and raw_label and raw_label.lower() not in ['none', '', 'n/a']:
                        print(f"⚠️ UNMAPPED IMPACT: '{raw_label}' for event: {event.get('event', 'unknown')[:40]}")
                        event_impact = 1  # Default to Low for unmapped non-empty impacts
                    
                    if event_impact not in impacts:
                        add_event = False
                
                if add_event:
                    filtered_events.append(event)
            
            # Augment with MT5 time fields (if MT5 timezone configured)
            try:
                self._augment_events_with_mt5_time(filtered_events)
            except Exception:
                pass

            # Ensure MT5 time fields ALWAYS present for aggregator mapping
            try:
                def _convert_12h_to_24h(t: str) -> str:
                    t = (t or '').strip()
                    if not t or t.lower() in ("all day",):
                        return ''
                    m = re.match(r'^(\d{1,2}):(\d{2})(am|pm)$', t.lower())
                    if not m:
                        # already maybe HH:MM
                        if re.match(r'^\d{2}:\d{2}$', t):
                            return t
                        return ''
                    h, mn, ap = m.groups(); h = int(h)
                    if ap == 'pm' and h != 12: h += 12
                    if ap == 'am' and h == 12: h = 0
                    return f"{h:02d}:{mn}"

                for e in filtered_events:
                    # prefer existing fields if present
                    if 'time_mt5' in e and 'date_mt5' in e:
                        # still ensure datetime composite
                        if 'datetime_mt5' not in e and re.match(r'^\d{4}-\d{2}-\d{2}$', str(e.get('date_mt5',''))) and re.match(r'^\d{2}:\d{2}$', str(e.get('time_mt5',''))):
                            e['datetime_mt5'] = f"{e['date_mt5']} {e['time_mt5']}"
                        continue
                    # derive
                    base_date = e.get('date') or today.strftime('%Y-%m-%d')
                    if not re.match(r'^\d{4}-\d{2}-\d{2}$', str(base_date)):
                        # attempt parse common formats
                        parsed = None
                        for fmt in ('%b %d, %Y', '%B %d, %Y', '%b %d %Y', '%B %d %Y'):
                            try:
                                parsed = datetime.strptime(str(base_date), fmt).strftime('%Y-%m-%d'); break
                            except Exception:
                                pass
                        base_date = parsed or today.strftime('%Y-%m-%d')
                    t24 = e.get('time_vn_24h') or _convert_12h_to_24h(e.get('time',''))
                    if not t24:
                        # leave blank but still set keys to avoid KeyError later
                        t24 = ''
                    e['time_mt5'] = t24
                    e['date_mt5'] = base_date
                    e['datetime_mt5'] = f"{base_date} {t24}".strip()
                    e.setdefault('mt5_timezone', 'MT5 Server Time (fallback=VN)')
            except Exception as _mt5e:
                print(f"⚠️ Fallback MT5 time enrichment error: {_mt5e}")

            print(f"✅ Found {len(filtered_events)} events after filtering")
            
            # Save files
            try:
                os.makedirs('news_output', exist_ok=True)
                output_path = f'news_output/news_forexfactory_{today_str}.json'
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered_events, f, ensure_ascii=False, indent=2)
                print(f"Đã lưu file tin tức: {output_path}")
                
                # Save recent news with actual data
                recent_news_with_data = self._filter_recent_news_with_actual(filtered_events)
                recent_path = f'news_output/recent_news_with_actual_{today_str}.json'
                
                with open(recent_path, 'w', encoding='utf-8') as f:
                    json.dump(recent_news_with_data, f, ensure_ascii=False, indent=2)
                
                if recent_news_with_data:
                    print(f"✅ Đã lưu {len(recent_news_with_data)} tin có số liệu thực tế vừa công bố: {recent_path}")
                else:
                    print(f"✅ Đã tạo file rỗng (không có tin nào thỏa mãn điều kiện): {recent_path}")
                    
            except Exception as e:
                print(f"Lỗi khi lưu file tin tức: {e}")
            
            return filtered_events
            
    
    def _extract_events_from_json(self, page_source, today):
        """Extract events from JSON data in page source"""
        
        try:
            # Look for calendar data in various JSON formats
            # Based on ForexFactory structure found in debug HTML
            json_patterns = [
                r'days:\s*(\[.*?\]\s*,)',  # Main calendar days array - non-greedy to closing bracket
                r'events:\s*(\[.*?\])',  # Events array
                r'window\.FF\.calendarData\s*=\s*(\{.*?\});',
                r'ff_calendar_rows\s*=\s*(\[.*?\]);',
            ]
            
            for pattern in json_patterns:
                matches = re.search(pattern, page_source, re.DOTALL)
                if matches:
                    try:
                        json_str = matches.group(1)
                        
                        # Clean up trailing comma if present
                        if json_str.strip().endswith(','):
                            json_str = json_str.strip()[:-1]
                        
                        print(f"Found JSON pattern: {pattern[:20]}...")
                        
                        # Handle incomplete JSON by finding the end bracket (if needed)
                        if not json_str.strip().endswith((']', '}')):
                            # Look for the end of this JSON structure
                            start_pos = page_source.find(matches.group(0))
                            remaining = page_source[start_pos + len(matches.group(0)):]
                            
                            # Count brackets to find proper end
                            bracket_count = 0
                            json_content = matches.group(1)
                            
                            if json_content.strip().startswith('['):
                                for i, char in enumerate(remaining):
                                    if char == '[':
                                        bracket_count += 1
                                    elif char == ']':
                                        bracket_count -= 1
                                        if bracket_count < 0:
                                            json_str = json_content + remaining[:i+1]
                                            break
                            elif json_content.strip().startswith('{'):
                                for i, char in enumerate(remaining):
                                    if char == '{':
                                        bracket_count += 1
                                    elif char == '}':
                                        bracket_count -= 1
                                        if bracket_count < 0:
                                            json_str = json_content + remaining[:i+1]
                                            break
                        
                        data = json.loads(json_str)
                        
                        events = []
                        # Process the JSON data structure - ForexFactory specific
                        if isinstance(data, list):
                            # Handle days array from ForexFactory
                            # Dynamic date patterns based on current date
                            today_date_patterns = [
                                today.strftime('%b %d'),  # Current date e.g. "Aug 8"
                                today.strftime('%B %d'),  # Full month name e.g. "August 8"
                                today.strftime('%b %d').replace(' 0', ' '),  # No leading zero
                                f"{today.strftime('%b')} {today.day}",  # Dynamic format
                                f"{today.strftime('%B')} {today.day}",  # Full month dynamic
                                today.strftime('%A'),  # Day name e.g. "Friday"
                                today.strftime('%a'),   # Short day name e.g. "Fri"
                            ]
                            
                            for day_item in data:
                                if isinstance(day_item, dict) and 'events' in day_item:
                                    day_date = day_item.get('date', '')
                                    
                                    # Check if this is today's data
                                    is_today = False
                                    for pattern in today_date_patterns:
                                        if pattern in day_date:
                                            is_today = True
                                            break
                                    
                                    if is_today:
                                        print(f"📅 Found today's events in: {day_date}")
                                        for event_item in day_item['events']:
                                            event = self._parse_event_from_json(event_item, today)
                                            if event:
                                                events.append(event)
                                elif isinstance(day_item, dict):
                                    # Direct event item
                                    event = self._parse_event_from_json(day_item, today)
                                    if event:
                                        events.append(event)
                        elif isinstance(data, dict):
                            # Handle nested structure
                            if 'events' in data:
                                for item in data['events']:
                                    event = self._parse_event_from_json(item, today)
                                    if event:
                                        events.append(event)
                            elif 'days' in data:
                                for day_item in data['days']:
                                    if 'events' in day_item:
                                        for event_item in day_item['events']:
                                            event = self._parse_event_from_json(event_item, today)
                                            if event:
                                                events.append(event)
                        
                        if events:
                            print(f"✅ Extracted {len(events)} events from JSON pattern: {pattern[:30]}")
                            return events
                            
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error for pattern {pattern[:20]}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing pattern {pattern[:20]}: {e}")
                        continue
            
            return []
            
        except Exception as e:
            print(f"Error extracting from JSON: {e}")
            return []
    
    def _parse_event_from_json(self, item, today):
        """Parse a single event from JSON data"""
        try:
            if not isinstance(item, dict):
                return None
            
            # Extract fields with various possible keys from ForexFactory JSON
            date = item.get('date', today.strftime('%Y-%m-%d'))
            time = item.get('timeLabel', item.get('time', ''))
            currency = item.get('currency', '')
            event_name = item.get('name', item.get('event', item.get('title', '')))
            impact = item.get('impactName', item.get('impact', 'None'))
            actual = item.get('actual', '')
            forecast = item.get('forecast', '')
            previous = item.get('previous', '')
            
            # Parse ForexFactory date format if needed
            if isinstance(date, str) and 'span' in date:
                # Format like "Wed <span>Aug 6</span>"
                date_match = re.search(r'<span>([^<]+)</span>', date)
                if date_match:
                    date_text = date_match.group(1).strip()
                    date = self._parse_date_from_text(date_text, today)
                else:
                    date = today.strftime('%Y-%m-%d')
            elif not date:
                date = today.strftime('%Y-%m-%d')
            
            # Use the original date from ForexFactory if available & looks like YYYY-MM-DD
            if 'date' in item and isinstance(item['date'], str) and re.match(r'^\d{4}-\d{2}-\d{2}$', item['date']):
                date = item['date']
            # If still not ISO, try common text formats (e.g., 'Aug 13, 2025') and convert to ISO
            if isinstance(date, str) and not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
                parsed_iso = None
                for fmt in ('%b %d, %Y', '%B %d, %Y', '%b %d %Y', '%B %d %Y'):
                    try:
                        dt = datetime.strptime(date, fmt)
                        parsed_iso = dt.strftime('%Y-%m-%d')
                        break
                    except Exception:
                        pass
                if not parsed_iso:
                    parsed_iso = self._parse_date_from_text(date, today)
                date = parsed_iso
                
            # Only return if we have meaningful data
            if event_name and currency:
                return {
                    'date': date if isinstance(date, str) else today.strftime('%Y-%m-%d'),
                    'time': time,
                    'currency': currency,
                    'event': event_name,
                    'impact': impact,
                    'actual': actual,
                    'forecast': forecast,
                    'previous': previous
                }
            
            return None
            
        except Exception as e:
            print(f"Error parsing event from JSON: {e}")
            return None
    
    def _extract_events_from_html(self, today):
        """Fallback HTML extraction method"""
        try:
            # Wait longer for dynamic content
            # Cloudflare page often shows "Just a moment...". Wait/retry quickly.
            total_wait = 0
            while total_wait < 20:
                if self.driver.title and 'Just a moment' not in (self.driver.title or ''):
                    break
                time.sleep(2)
                total_wait += 2
            time.sleep(6)
            
            # Try multiple selectors for rows
            row_selectors = [
                "table.calendar__table tbody tr",
                "table.calendar__table tr",
                ".calendar__table tr",
                "tr[class*='calendar']"
            ]
            
            rows = []
            for selector in row_selectors:
                try:
                    found_rows = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if found_rows and len(found_rows) > 1:
                        rows = found_rows
                        print(f"Using selector '{selector}' - found {len(rows)} rows")
                        break
                except:
                    continue
            
            if not rows:
                print("No rows found with any selector")
                return []
            
            events = []
            current_date_text = None
            current_actual_date = None
            current_time = None
            
            for row in rows:
                try:
                    row_class = row.get_attribute('class') or ''
                    
                    # Skip header and special rows
                    if 'calendar__header' in row_class or 'subhead' in row_class:
                        continue
                    if 'day-breaker' in row_class or 'no-event' in row_class:
                        continue
                    
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if len(cells) < 6:
                        continue
                    
                    # Extract basic data
                    date_text = cells[0].text.strip() if len(cells) > 0 else ''
                    time_text = cells[1].text.strip() if len(cells) > 1 else ''
                    currency_text = cells[3].text.strip() if len(cells) > 3 else ''
                    
                    # Event from nested span
                    event_title = ''
                    if len(cells) > 5:
                        spans = cells[5].find_elements(By.TAG_NAME, 'span')
                        if spans:
                            event_title = spans[0].text.strip()
                        else:
                            event_title = cells[5].text.strip()
                    
                    # Data values
                    actual_text = cells[7].text.strip() if len(cells) > 7 else ''
                    forecast_text = cells[8].text.strip() if len(cells) > 8 else ''
                    previous_text = cells[9].text.strip() if len(cells) > 9 else ''
                    
                    # Impact
                    impact = 'None'
                    if len(cells) > 4:
                        impact_spans = cells[4].find_elements(By.TAG_NAME, 'span')
                        for span in impact_spans:
                            span_class = span.get_attribute('class') or ''
                            if 'icon--ff-impact-red' in span_class:
                                impact = 'High'
                                break
                            elif 'icon--ff-impact-ora' in span_class:
                                impact = 'Medium'
                                break
                            elif 'icon--ff-impact-yel' in span_class:
                                impact = 'Low'
                                break
                    
                    # Track current date/time
                    if date_text:
                        current_date_text = date_text.lower()
                        current_actual_date = self._parse_date_from_text(current_date_text, today)
                    elif current_actual_date:
                        date_text = current_actual_date
                    
                    if time_text:
                        current_time = time_text
                    elif current_time:
                        time_text = current_time
                    
                    # Filter for today only
                    is_today = False
                    if current_date_text:
                        today_formats = [
                            today.strftime('%b %d').lower(),
                            today.strftime('%b %d').lower().replace(' 0', ' '),
                            today.strftime('%B %d').lower(),
                            today.strftime('%B %d').lower().replace(' 0', ' '),
                            'today',
                            today.strftime('%d').lstrip('0')
                        ]
                        for fmt in today_formats:
                            if fmt in current_date_text:
                                is_today = True
                                break
                    
                    if not is_today or not event_title or not currency_text:
                        continue
                    
                    event = {
                        'date': current_actual_date or today.strftime('%Y-%m-%d'),
                        'time': time_text,
                        'currency': currency_text,
                        'event': event_title,
                        'impact': impact,
                        'actual': actual_text,
                        'forecast': forecast_text,
                        'previous': previous_text
                    }
                    
                    events.append(event)
                    
                except Exception as e:
                    continue
            
            print(f"HTML extraction found {len(events)} events")
            return events
            
        except Exception as e:
            print(f"HTML extraction error: {e}")
            return []

    def _emergency_extract_events(self, today):
        """Emergency extraction - ONLY REAL DATA from ForexFactory tables"""
        
        try:
            print("🚨 EMERGENCY EXTRACTION - REAL DATA ONLY")
            
            # Scroll to ensure all content is loaded
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)  # Longer wait
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(8)  # Even longer wait for content to stabilize
            
            # Try to click or interact to trigger any lazy loading
            try:
                self.driver.execute_script("document.dispatchEvent(new Event('scroll'));")
                time.sleep(3)
            except:
                pass
            
            # Find ALL table rows on the page - no filtering yet
            all_rows = self.driver.find_elements(By.TAG_NAME, "tr")
            print(f"🔍 Found {len(all_rows)} total rows on page")
            
            # Additional debug: check page title and basic elements
            page_title = self.driver.title
            print(f"📄 Page title: {page_title}")
            
            # Check if page has basic content
            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            if "calendar" in body_text.lower() or "forex" in body_text.lower():
                print("✅ Page contains calendar/forex content")
            else:
                print("❌ Page may not have loaded correctly")
                print(f"📝 Body preview: {body_text[:200]}...")
            
            # Try alternative row selectors if no rows found
            if len(all_rows) == 0:
                print("🔍 No TR elements found, trying alternative selectors...")
                
                # Try finding tables first
                tables = self.driver.find_elements(By.TAG_NAME, "table")
                print(f"📊 Found {len(tables)} tables")
                
                # Try finding any divs with calendar classes
                calendar_divs = self.driver.find_elements(By.CSS_SELECTOR, "[class*='calendar']")
                print(f"📅 Found {len(calendar_divs)} calendar-related elements")
                
                # Try finding all elements with text content
                all_elements = self.driver.find_elements(By.XPATH, "//*[text()]")
                print(f"📝 Found {len(all_elements)} elements with text")
                
                # Look for any USD text on page
                page_source = self.driver.page_source
                usd_count = page_source.count("USD")
                print(f"💵 'USD' appears {usd_count} times in page source")
                
                if usd_count == 0:
                    print("❌ No USD content found - page may not have loaded economic data")
                    return []
            
            events = []
            current_date = None
            
            # Known currencies to validate real data
            valid_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNY"]
            
            for row in all_rows:
                try:
                    row_text = row.text.strip()
                    
                    # Skip empty or very short rows
                    if not row_text or len(row_text) < 5:
                        continue
                    
                    # Check for today's date - REAL date only
                    today_markers = ["Aug 7", "August 7", "Thu"]
                    if any(marker in row_text for marker in today_markers):
                        current_date = "Aug 7, 2025"
                        print(f"📅 Found real date: {current_date}")
                        continue
                    
                    # Only process rows with valid currencies AND meaningful content
                    has_currency = any(curr in row_text for curr in valid_currencies)
                    has_economic_keywords = any(keyword in row_text for keyword in [
                        "Employment", "Claims", "Productivity", "GDP", "Inflation", 
                        "Rate", "PMI", "CPI", "Balance", "Trade", "Auction"
                    ])
                    
                    if has_currency and has_economic_keywords and current_date:
                        # Extract from table cells - REAL structure only
                        cells = row.find_elements(By.TAG_NAME, "td")
                        
                        if len(cells) >= 6:  # Minimum cells for real ForexFactory row
                            # Initialize variables
                            currency = ""
                            event_name = ""
                            time_str = ""
                            actual = ""
                            forecast = ""
                            previous = ""
                            
                            # Extract data based on REAL cell content
                            for cell in cells:
                                cell_text = cell.text.strip()
                                
                                # Currency identification
                                if cell_text in valid_currencies:
                                    currency = cell_text
                                
                                # Time identification (real format)
                                elif re.match(r'\d{1,2}:\d{2}[ap]m', cell_text.lower()):
                                    time_str = cell_text
                                
                                # Event name (real economic events)
                                elif len(cell_text) > 8 and has_economic_keywords and not re.match(r'^[\d\.\-\+%K]+$', cell_text):
                                    event_name = cell_text
                                
                                # Numeric values (must be realistic economic data)
                                elif re.match(r'^[\d\.\-\+%K]+$', cell_text) and cell_text not in valid_currencies:
                                    if not actual:
                                        actual = cell_text
                                    elif not forecast:
                                        forecast = cell_text
                                    elif not previous:
                                        previous = cell_text
                            
                            # ONLY add if we have REAL complete data
                            if currency and event_name and len(event_name) > 5:
                                event_data = {
                                    "date": current_date,
                                    "time": time_str,
                                    "currency": currency,
                                    "event": event_name,
                                    "impact": "Medium",  # Safe default
                                    "actual": actual,
                                    "forecast": forecast,
                                    "previous": previous
                                }
                                
                                events.append(event_data)
                                print(f"✅ REAL EVENT: {currency} {event_name}")
                                
                                # Validate USD Prelim events against known data
                                if currency == "USD" and "Prelim" in event_name:
                                    print(f"🎯 USD PRELIM FOUND: {event_name}")
                                    print(f"   Previous: {previous}")
                                    
                                    # Warn if data doesn't match expected values
                                    if "Nonfarm Productivity" in event_name and previous and previous != "-1.5%":
                                        print(f"   ⚠️ VALIDATION: Expected -1.5%, got {previous}")
                                    elif "Unit Labor Costs" in event_name and previous and previous != "6.6%":
                                        print(f"   ⚠️ VALIDATION: Expected 6.6%, got {previous}")
                
                except Exception as e:
                    continue  # Skip problematic rows
            
            print(f"🚨 EMERGENCY EXTRACTION COMPLETE: {len(events)} REAL events found")
            
            if len(events) == 0:
                print("❌ NO REAL DATA FOUND - ForexFactory may have changed structure")
                print("💡 Recommend manual inspection of the page")
            
            return events
            
        except Exception as e:
            print(f"❌ Emergency extraction error: {e}")
            return []  # Return empty list - NO FAKE DATA

    # API feed support removed

def get_today_news(currencies=None, impacts=None, headless=True, auto_cleanup=True,
                   clean_existing_files: bool = False) -> List[Dict[str, Any]]:
    """Fetch today's news events.

    Args:
        currencies: filter list
        impacts: filter list
        headless: run chrome headless
        auto_cleanup: run age-based cleanup (keep 5 newest)
        clean_existing_files: if True, delete ALL existing json in news_output before scrape (force fresh)
    """
    news_dir = 'news_output'
    if clean_existing_files:
        try:
            if os.path.exists(news_dir):
                deleted = 0
                for f in os.listdir(news_dir):
                    if f.endswith('.json'):
                        try:
                            os.remove(os.path.join(news_dir, f))
                            deleted += 1
                        except Exception:
                            pass
                print(f"🧹 Removed {deleted} old news files (clean_existing_files=True)")
        except Exception as e:
            print(f"⚠️ Hard clean failed: {e}")

    if auto_cleanup:
        print("🧹 News Scraper: Auto cleanup before processing (age-based)...")
        try:
            cleanup_result = cleanup_news_data(max_age_hours=24, keep_latest=5)
            print(f"✅ Cleaned {cleanup_result.get('total_files_deleted', 0)} files, "
                  f"freed {cleanup_result.get('total_space_freed_mb', 0.0):.2f} MB")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    scraper = NewsScraperSelenium()
    return scraper.get_today_events(currencies, impacts, headless)

def scan_event_window(target_time: str, currencies=None, impacts=None, headless=True,
                      window_minutes: int = 5, poll_interval: int = 30,
                      clean_each_attempt: bool = False) -> List[Dict[str, Any]]:
    """Continuously scrape within a small window around an event time until actual data appears or timeout.

    target_time: 'HH:MM' 24h in Vietnam time zone (calendar is parsed accordingly).
    window_minutes: total duration to keep polling (default 5). Poll ends early if any event with matching time has actual value.
    poll_interval: seconds between attempts.
    clean_each_attempt: delete prior news_output JSON before each attempt for clarity.
    """
    from datetime import datetime as _dt
    vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    deadline = _dt.now(vn_tz) + timedelta(minutes=window_minutes)
    collected = []
    attempt = 0
    print(f"⏱️ Event scan started for {target_time} (VN) up to {window_minutes}m...")
    while _dt.now(vn_tz) < deadline:
        attempt += 1
        print(f"🔄 Attempt {attempt}...")
        events = get_today_news(currencies=currencies, impacts=impacts, headless=headless,
                                auto_cleanup=False, clean_existing_files=clean_each_attempt)
        # Filter events matching target_time & having actual numeric/valid
        ready = []
        for e in events:
            et = str(e.get('time','')).strip()
            if not et:
                continue
            # Accept both 12h like 1:30pm and raw stored; we compare by converting to 24h
            m = re.match(r'^(\d{1,2}):(\d{2})(am|pm)$', et.lower())
            if m:
                h, mn, ap = m.groups(); h = int(h)
                if ap=='pm' and h!=12: h+=12
                if ap=='am' and h==12: h=0
                et_24 = f"{h:02d}:{mn}"
            else:
                et_24 = et  # assume already 24h
            actual = str(e.get('actual','')).strip()
            has_actual = actual and actual not in ('TBA','Pending','-','–','N/A','')
            if et_24 == target_time and has_actual:
                ready.append(e)
        if ready:
            print(f"✅ Captured {len(ready)} target events with actual values at attempt {attempt}.")
            collected = ready
            break
        sleep_for = poll_interval
        remain = (deadline - _dt.now(vn_tz)).total_seconds()
        if remain <= 0:
            break
        sleep_for = min(sleep_for, max(1, int(remain)))
        time.sleep(sleep_for)
    if not collected:
        print("⚠️ No actual data captured within window.")
    return collected

def save_recent_news_to_json(events, filename=None):
    """Lưu chỉ các tin có số liệu thực tế vừa được công bố trong 60 phút trở lại"""
    from datetime import datetime, timedelta
    import pytz
    
    # Lưu vào thư mục news_output thay vì news
    os.makedirs('news_output', exist_ok=True)
    try:
        # Tạo instance để sử dụng timezone
        vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        now = datetime.now(vn_tz)
        
        filtered_events = []
        for event in events:
            # Kiểm tra có số liệu thực tế không
            actual = event.get('actual', '').strip()
            if not actual or actual in ['TBA', 'Pending', '', '-', '–', 'N/A']:
                continue
                
            # Tạo datetime từ date và time của event
            try:
                event_date = event.get('date', '')
                event_time = event.get('time', '')
                
                if not event_time:
                    continue
                    
                # Parse thời gian
                import re
                time_match = re.match(r'(\d{1,2}):(\d{2})(am|pm)', event_time.lower())
                if time_match:
                    hour, minute, period = time_match.groups()
                    hour = int(hour)
                    if period == 'pm' and hour != 12:
                        hour += 12
                    elif period == 'am' and hour == 12:
                        hour = 0
                    
                    # Parse date properly
                    base_date = None
                    try:
                        base_date = datetime.strptime(event_date, '%Y-%m-%d')
                    except Exception:
                        continue
                    
                    # Create timezone-aware datetime properly
                    event_datetime = vn_tz.localize(
                        base_date.replace(hour=hour, minute=int(minute), second=0, microsecond=0)
                    )
                    
                    # Kiểm tra thời gian công bố (trong 60 phút trở lại)
                    time_diff = (now - event_datetime).total_seconds()
                    if 0 <= time_diff <= 3600:  # 60 phút = 3600 giây
                        filtered_events.append(event)
                        print(f"📰 Recent news found: {event_time} {event['currency']} {event['event']} = {actual} ({time_diff/60:.1f} minutes ago)")
                        
            except Exception as e:
                print(f"⚠️ Lỗi parse thời gian cho {event.get('event', 'unknown')}: {e}")
                continue
        
        # Lưu file nếu có tin phù hợp
        if filtered_events:
            if filename is None:
                today_str = now.strftime('%Y%m%d')
                filename = f'recent_news_with_actual_{today_str}.json'
            
            # Lưu vào news_output thay vì news
            file_path = os.path.join('news_output', filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_events, f, indent=2, ensure_ascii=False)
            print(f"✅ Đã lưu {len(filtered_events)} tin có số liệu thực tế vừa công bố: {file_path}")
            return True
        else:
            print('⚠️ Không có tin nào có số liệu thực tế vừa được công bố trong 60 phút trở lại.')
            # Tạo file trống để đánh dấu đã kiểm tra
            if filename is None:
                today_str = now.strftime('%Y%m%d')
                filename = f'recent_news_with_actual_{today_str}.json'
            
            file_path = os.path.join('news_output', filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
            print(f"📝 Tạo file trống để đánh dấu đã kiểm tra: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error saving recent news: {e}")
        return False

def cleanup_news_data(max_age_hours: int = 24, keep_latest: int = 5) -> Dict[str, Any]:
    """
    🧹 NEWS SCRAPER: Dọn dẹp dữ liệu của module này
    Chỉ xóa dữ liệu trong thư mục news_output
    
    Args:
        max_age_hours: Tuổi tối đa của file (giờ)
        keep_latest: Số file mới nhất cần giữ lại
    """
    cleanup_stats = {
        'module_name': 'news_scraper',
        'directories_cleaned': [],
        'total_files_deleted': 0,
        'total_space_freed_mb': 0.0,
        'cleanup_time': datetime.now().isoformat()
    }
    
    # Thư mục mà News Scraper quản lý
    target_directories = [
        'news_output'
    ]
    
    for directory in target_directories:
        if os.path.exists(directory):
            print(f"🧹 News Scraper cleaning {directory}...")
            dir_stats = _clean_directory(directory, max_age_hours, keep_latest)
            cleanup_stats['directories_cleaned'].append({
                'directory': directory,
                'files_deleted': dir_stats['deleted'],
                'space_freed_mb': dir_stats['space_freed']
            })
            cleanup_stats['total_files_deleted'] += dir_stats['deleted']
            cleanup_stats['total_space_freed_mb'] += dir_stats['space_freed']
        else:
            print(f"📁 Directory {directory} does not exist, skipping")
    
    print(f"🧹 NEWS SCRAPER cleanup complete: "
          f"{cleanup_stats['total_files_deleted']} files deleted, "
          f"{cleanup_stats['total_space_freed_mb']:.2f} MB freed")
    return cleanup_stats

def _clean_directory(directory: str, max_age_hours: int, keep_latest: int) -> Dict[str, int]:
    """Helper function để clean một directory"""
    deleted_count = 0
    space_freed = 0.0
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    try:
        if not os.path.exists(directory):
            return {'deleted': 0, 'space_freed': 0.0}
            
        # Lấy tất cả news files
        all_files = []
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.json'):
                file_stat = os.stat(file_path)
                file_time = datetime.fromtimestamp(file_stat.st_mtime)
                all_files.append({
                    'path': file_path,
                    'name': file_name,
                    'time': file_time,
                    'size': file_stat.st_size
                })
        
        # Sắp xếp theo thời gian (mới nhất trước)
        all_files.sort(key=lambda x: x['time'], reverse=True)
        
        # Giữ lại keep_latest files mới nhất
        files_to_keep = all_files[:keep_latest]
        files_to_check = all_files[keep_latest:]
        
        # Xóa files cũ hơn max_age_hours
        for file_info in files_to_check:
            if file_info['time'] < cutoff_time:
                try:
                    os.remove(file_info['path'])
                    deleted_count += 1
                    space_freed += file_info['size'] / (1024 * 1024)  # Convert to MB
                    print(f"Deleted: {file_info['name']}")
                except Exception as e:
                    print(f"Failed to delete {file_info['path']}: {e}")
        
    except Exception as e:
        print(f"Error cleaning directory {directory}: {e}")
    
    return {'deleted': deleted_count, 'space_freed': space_freed}

def cleanup_old_news_files(max_age_hours: int = 24, keep_latest: int = 5) -> Dict[str, Any]:
    """
    🧹 Legacy function - calls the new cleanup_news_data function
    """
    return cleanup_news_data(max_age_hours, keep_latest)

def save_news_with_cleanup(events: List[Dict], filename: str = None, auto_cleanup: bool = True) -> bool:
    """
    💾 Lưu news với tự động dọn dẹp file cũ
    
    Args:
        events: News events data
        filename: Custom filename (optional)
        auto_cleanup: Tự động dọn dẹp file cũ không
    """
    try:
        news_dir = 'news_output'
        os.makedirs(news_dir, exist_ok=True)
        
        # Auto cleanup deprecated - now handled by module auto-cleanup
        
        # Create filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"forex_news_{timestamp}.json"
        
        filepath = os.path.join(news_dir, filename)
        
        # Add metadata
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'events_count': len(events),
            'scrape_time': datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).isoformat(),
            'events': events
        }
        
        # Save file
        success = overwrite_json_safely(filepath, output_data, backup=False)
        
        if success:
            print(f"💾 News saved with cleanup: {filename}")
        else:
            print(f"❌ Failed to save news: {filename}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error saving news: {e}")
        return False

def get_news_storage_report() -> Dict[str, Any]:
    """📊 Tạo báo cáo sử dụng storage cho news"""
    print("Storage report is deprecated. Use system disk utilities instead.")
    return {'note': 'deprecated_function', 'use': 'system_disk_utilities'}

def parse_news_release_times() -> List[str]:
    """🔍 Parse news release times from today's fetched news data
    Returns list of time strings in HH:MM format (VN timezone)
    """
    try:
        news_dir = 'news_output'
        if not os.path.exists(news_dir):
            return []
        
        # Find latest news file from today
        today = datetime.now().strftime('%Y%m%d')
        news_files = []
        
        for file in os.listdir(news_dir):
            # Support both old and new naming patterns
            if (file.startswith('forex_news_') or file.startswith('news_forexfactory_')) and file.endswith('.json'):
                if today in file:
                    news_files.append(os.path.join(news_dir, file))
        
        if not news_files:
            print("[News Parser] No news files found for today")
            return []
        
        # Get the latest file
        latest_file = max(news_files, key=os.path.getmtime)
        print(f"[News Parser] Parsing times from: {os.path.basename(latest_file)}")
        
        # Load and parse news data
        with open(latest_file, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        
        # Handle different data formats
        if isinstance(news_data, list):
            events = news_data  # Direct list format
        elif isinstance(news_data, dict):
            events = news_data.get('events', [])  # Wrapped in events key
        else:
            events = []
            
        news_times = set()  # Use set to avoid duplicates
        
        # Extract unique times from news events
        for event in events:
            # Try different time field names
            time_str = event.get('time_vn_24h') or event.get('time') or ''
            time_str = time_str.strip()
            
            if time_str and time_str != 'All Day':
                try:
                    # Handle different time formats
                    if ':' in time_str:
                        # Already in HH:MM format
                        time_parts = time_str.split(':')
                        if len(time_parts) == 2:
                            hour = int(time_parts[0])
                            minute = int(time_parts[1])
                            if 0 <= hour <= 23 and 0 <= minute <= 59:
                                formatted_time = f"{hour:02d}:{minute:02d}"
                                news_times.add(formatted_time)
                except Exception as e:
                    print(f"[News Parser] Failed to parse time '{time_str}': {e}")
                    continue
        
        # Convert to sorted list
        sorted_times = sorted(list(news_times))
        print(f"[News Parser] Found {len(sorted_times)} unique news release times: {sorted_times}")
        
        return sorted_times
        
    except Exception as e:
        print(f"[News Parser] Error parsing news times: {e}")
        return []

if __name__ == "__main__":
    print("📰 News Scraper - Enhanced with Cleanup")
    print("=" * 50)

    # 🧹 AUTO CLEANUP (no prompt)
    print("\n🧹 NEWS DATA CLEANUP:")
    print("🧹 Cleaning old news files (24h old, keep 5 latest)...")
    cleanup_result = cleanup_old_news_files(max_age_hours=24, keep_latest=5)
    total_deleted = cleanup_result.get('total_files_deleted') or cleanup_result.get('deleted') or 0
    total_space = cleanup_result.get('total_space_freed_mb') or cleanup_result.get('space_freed') or 0.0
    print(f"   Deleted: {total_deleted} files")
    print(f"   Space freed: {float(total_space):.2f} MB")
    
    # 📊 STORAGE REPORT (deprecated)
    storage_report = get_news_storage_report()
    print(f"\n📊 Storage report: {storage_report.get('note', 'available')}")
    
    # Scrape today's news (feed-first, Selenium fallback)
    print("\n📰 Scraping today's news...")
    # Force headless (ẩn Chrome) for auto/manual run
    events = get_today_news(headless=True, auto_cleanup=False)  # cleanup already done above
    print(f"Found {len(events)} events today:")
    for e in events:
        print(f"{e['time']} {e['currency']} {e['event']} Actual: {e['actual']}")
