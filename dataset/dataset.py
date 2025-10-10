import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_thai_churn_dataset(n_customers=100000):
    """
    Generate realistic customer churn dataset for Thai telecom/internet provider
    Based on Thai market characteristics, demographics, and behavior patterns
    """
    
    print(f"🇹🇭 Generating {n_customers:,} Thai customer records...")
    
    # Thai-specific data
    thai_first_names_male = [
        'Somchai', 'Somsak', 'Somkid', 'Surasak', 'Somboon', 'Prasert', 'Suchart',
        'Wichai', 'Boonmee', 'Anucha', 'Chaiwat', 'Sompong', 'Manop', 'Narong',
        'Prayut', 'Thawatchai', 'Watchara', 'Pongsakorn', 'Natthaphon', 'Apirak',
        'Kritsada', 'Kittipong', 'Thanawat', 'Anurak', 'Pisit', 'Chatchai'
    ]
    
    thai_first_names_female = [
        'Somying', 'Sukanya', 'Wannee', 'Pranee', 'Sunee', 'Siriporn', 'Pensri',
        'Nittaya', 'Malee', 'Rattana', 'Orathai', 'Siriwan', 'Supaporn', 'Jintana',
        'Pimchanok', 'Chanida', 'Nittha', 'Suchada', 'Kulthida', 'Rattanaporn',
        'Waranya', 'Nantawan', 'Phimphakan', 'Kanokwan', 'Sutthida', 'Pawitra'
    ]
    
    thai_last_names = [
        'Saetang', 'Chaiwong', 'Srisawat', 'Pongpanit', 'Thongchai', 'Wongsawat',
        'Rattanakorn', 'Sombatpiboon', 'Kittikhun', 'Pornprasert', 'Wongcharoen',
        'Sukkasem', 'Ratanaporn', 'Pattanakul', 'Thienprasert', 'Mahawitthayalai',
        'Charoensuk', 'Phrakongsap', 'Sinthuphanthong', 'Suwannaphoom',
        'Rujirawat', 'Pattanaporn', 'Kittisak', 'Watthanawong', 'Suwannapoom'
    ]
    
    # Thai provinces (using major provinces)
    thai_provinces = [
        'Bangkok', 'Nonthaburi', 'Pathum Thani', 'Samut Prakan', 'Samut Sakhon',
        'Chiang Mai', 'Chiang Rai', 'Nakhon Ratchasima', 'Khon Kaen', 'Udon Thani',
        'Ubon Ratchathani', 'Songkhla', 'Phuket', 'Surat Thani', 'Nakhon Si Thammarat',
        'Ayutthaya', 'Chon Buri', 'Rayong', 'Nakhon Pathom', 'Ratchaburi',
        'Lopburi', 'Saraburi', 'Phitsanulok', 'Lampang', 'Prachuap Khiri Khan'
    ]
    
    # Province weights (Bangkok Metro area has more customers) - FIXED TO SUM TO 1.0
    province_weights = np.array([
        0.25, 0.08, 0.06, 0.05, 0.03,  # Bangkok Metro = 0.47
        0.04, 0.02, 0.04, 0.03, 0.03,  # North & Northeast = 0.16
        0.03, 0.03, 0.03, 0.03, 0.02,  # South = 0.14
        0.03, 0.04, 0.03, 0.03, 0.02,  # Central = 0.15
        0.02, 0.02, 0.02, 0.02, 0.02   # Others = 0.10
    ])
    # Normalize to ensure sum = 1.0
    province_weights = province_weights / province_weights.sum()
    
    # Thai postal codes by province (simplified - first 2 digits)
    province_postcodes = {
        'Bangkok': '10', 'Nonthaburi': '11', 'Pathum Thani': '12', 
        'Samut Prakan': '10', 'Samut Sakhon': '74',
        'Chiang Mai': '50', 'Chiang Rai': '57', 'Nakhon Ratchasima': '30',
        'Khon Kaen': '40', 'Udon Thani': '41', 'Ubon Ratchathani': '34',
        'Songkhla': '90', 'Phuket': '83', 'Surat Thani': '84',
        'Nakhon Si Thammarat': '80', 'Ayutthaya': '13', 'Chon Buri': '20',
        'Rayong': '21', 'Nakhon Pathom': '73', 'Ratchaburi': '70',
        'Lopburi': '15', 'Saraburi': '18', 'Phitsanulok': '65',
        'Lampang': '52', 'Prachuap Khiri Khan': '77'
    }
    
    # Thai telecom providers for switching scenarios
    thai_telcos = ['AIS', 'TrueMove', 'dtac', 'NT Mobile', 'TOT']
    
    # Thai payment methods (popular in Thailand)
    payment_methods = [
        'PromptPay', 'Credit Card', 'Bank Transfer', 'True Money Wallet',
        '7-Eleven Counter', 'Line Pay', 'K-Plus', 'Mobile Banking'
    ]
    payment_method_weights = np.array([0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05])
    payment_method_weights = payment_method_weights / payment_method_weights.sum()
    
    # Internet service types (Thai market)
    internet_types = ['Fiber', 'ADSL', 'Cable', '4G', '5G', 'None']
    
    data = []
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 10, 1)
    
    for i in range(n_customers):
        if i % 10000 == 0:
            print(f"Progress: {i:,}/{n_customers:,}")
        
        # ===== DEMOGRAPHICS (Thai-specific) =====
        gender = np.random.choice(['Male', 'Female'], p=[0.49, 0.51])
        
        # Thai naming convention
        if gender == 'Male':
            first_name = random.choice(thai_first_names_male)
        else:
            first_name = random.choice(thai_first_names_female)
        
        last_name = random.choice(thai_last_names)
        full_name = f"{first_name} {last_name}"
        
        # Age distribution (Thailand demographics - younger population)
        age = int(np.random.normal(38, 14))
        age = max(18, min(75, age))
        senior_citizen = 1 if age >= 60 else 0  # Thai retirement age
        
        # Location (weighted toward Bangkok)
        province = np.random.choice(thai_provinces, p=province_weights)
        postcode_prefix = province_postcodes[province]
        postcode = f"{postcode_prefix}{random.randint(100, 999)}"
        
        # Living situation (Thai context)
        partner = np.random.choice(['Yes', 'No'], p=[0.55, 0.45])  # Thai marriage rate
        dependents = np.random.choice([0, 1, 2, 3, 4], p=[0.25, 0.30, 0.25, 0.15, 0.05])
        
        # ===== ACCOUNT INFORMATION =====
        account_created = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        tenure_months = (end_date - account_created).days // 30
        
        # Contract types (Thai telecom market)
        contract_type = np.random.choice(
            ['Prepaid', 'Postpaid Monthly', '1 Year', '2 Year'],
            p=[0.40, 0.35, 0.15, 0.10]  # Thailand has high prepaid usage
        )
        
        # ===== SERVICES =====
        # Internet service (Thai market - high fiber penetration in cities)
        if province in ['Bangkok', 'Nonthaburi', 'Pathum Thani', 'Chiang Mai', 'Chon Buri']:
            internet_weights = np.array([0.45, 0.20, 0.15, 0.10, 0.10])
            internet_weights = internet_weights / internet_weights.sum()
            internet_service = np.random.choice(
                ['Fiber', '5G', '4G', 'Cable', 'None'],
                p=internet_weights
            )
        else:
            internet_weights = np.array([0.35, 0.25, 0.15, 0.10, 0.15])
            internet_weights = internet_weights / internet_weights.sum()
            internet_service = np.random.choice(
                ['4G', 'Fiber', 'ADSL', 'Cable', 'None'],
                p=internet_weights
            )
        
        has_internet = internet_service != 'None'
        
        # Add-on services
        online_security = np.random.choice([0, 1], p=[0.65, 0.35]) if has_internet else 0
        online_backup = np.random.choice([0, 1], p=[0.70, 0.30]) if has_internet else 0
        device_protection = np.random.choice([0, 1], p=[0.60, 0.40]) if has_internet else 0
        tech_support = np.random.choice([0, 1], p=[0.75, 0.25]) if has_internet else 0
        
        # Thai-specific streaming services
        has_netflix_thai = np.random.choice([0, 1], p=[0.70, 0.30]) if has_internet else 0
        has_line_tv = np.random.choice([0, 1], p=[0.65, 0.35]) if has_internet else 0
        has_true_visions = np.random.choice([0, 1], p=[0.75, 0.25]) if has_internet else 0
        
        # Multiple SIM cards (common in Thailand)
        multiple_lines = np.random.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.20, 0.05])
        
        # ===== BILLING (Thai Baht - THB) =====
        # Monthly charges in THB (realistic Thai prices)
        if contract_type == 'Prepaid':
            monthly_charges = np.random.choice([199, 299, 399, 499, 599, 799])
        elif internet_service == 'Fiber':
            monthly_charges = np.random.choice([590, 690, 890, 1190, 1590])
        elif internet_service == '5G':
            monthly_charges = np.random.choice([699, 899, 1099, 1299])
        else:
            monthly_charges = np.random.choice([399, 499, 599, 799, 999])
        
        # Add-on costs
        addon_cost = 0
        if online_security:
            addon_cost += 50
        if online_backup:
            addon_cost += 100
        if device_protection:
            addon_cost += 150
        if has_netflix_thai:
            addon_cost += 349  # Netflix Basic Thai price
        if has_line_tv:
            addon_cost += 159
        if has_true_visions:
            addon_cost += 399
        
        monthly_charges += addon_cost
        
        # Total charges
        total_charges = monthly_charges * tenure_months * np.random.uniform(0.95, 1.05)
        
        # Price changes (Thai telcos often have promotions then increase)
        if tenure_months > 6:
            price_increase_weights = np.array([0.65, 0.15, 0.10, 0.05, 0.03, 0.02])
            price_increase_weights = price_increase_weights / price_increase_weights.sum()
            price_increase_6months = np.random.choice(
                [0, 50, 100, 150, 200, 300],
                p=price_increase_weights
            )
        else:
            price_increase_6months = 0
        
        # Payment method (Thai-specific)
        payment_method = np.random.choice(payment_methods, p=payment_method_weights)
        
        # Paperless billing (Line notification is popular)
        paperless_billing = np.random.choice(['Yes', 'No'], p=[0.70, 0.30])
        
        # Payment delays (consider Thai payment culture)
        payment_delay_days = max(0, int(np.random.exponential(3)))
        num_payment_failures = np.random.poisson(0.5)
        
        # Discount applied (promotion culture in Thailand)
        has_discount = np.random.choice([0, 1], p=[0.60, 0.40])
        if has_discount:
            discount_amount = np.random.choice([50, 100, 150, 200, 300, 500])
        else:
            discount_amount = 0
        
        # ===== USAGE PATTERNS =====
        # Data usage (Thais are heavy mobile internet users)
        if internet_service in ['Fiber', '5G', '4G']:
            data_usage_gb = np.random.gamma(3, 30)  # Heavy usage
        elif internet_service in ['Cable', 'ADSL']:
            data_usage_gb = np.random.gamma(2, 20)
        else:
            data_usage_gb = 0
        
        # Call minutes (decreasing due to Line/WhatsApp usage)
        call_minutes = int(np.random.gamma(2, 80))
        
        # SMS (very low due to Line dominance in Thailand)
        sms_count = np.random.poisson(10)
        
        # Line/Social media usage hours (very high in Thailand)
        social_media_hours = np.random.gamma(4, 30)
        
        # App usage (provider app)
        app_usage_hours = np.random.gamma(1.5, 2)
        last_login_days_ago = int(np.random.exponential(15))
        
        # Website visits (self-service portal)
        website_visits = np.random.poisson(3)
        
        # Roaming (Thai people travel to nearby countries)
        international_roaming = np.random.choice([0, 1], p=[0.85, 0.15])
        if international_roaming:
            roaming_charges = np.random.gamma(2, 150)
        else:
            roaming_charges = 0
        
        # ===== CUSTOMER SERVICE =====
        # Support tickets (Thai customer service expectations)
        num_support_tickets = np.random.poisson(2)
        num_support_tickets_6months = min(num_support_tickets, np.random.poisson(1.2))
        
        # Resolution time (Thai working hours consideration)
        avg_ticket_resolution_hours = np.random.gamma(2, 12)
        
        # Complaints
        num_complaints = np.random.poisson(0.3)
        
        # Customer satisfaction (1-5 scale, Thai tend to give high scores)
        customer_satisfaction = np.random.beta(9, 2) * 4 + 1
        
        # NPS Score
        nps_score = int(np.random.normal(25, 35))
        nps_score = max(-100, min(100, nps_score))
        
        # Support channel (Line is dominant in Thailand)
        support_channel_weights = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05])
        support_channel_weights = support_channel_weights / support_channel_weights.sum()
        support_channel = np.random.choice(
            ['Line Chat', 'Phone', 'Facebook Messenger', 'Walk-in Store', 'Email', 'App'],
            p=support_channel_weights
        )
        
        # ===== PRODUCT CHANGES =====
        num_plan_changes = np.random.poisson(1.5)
        plan_downgrade_6months = np.random.choice([0, 1], p=[0.80, 0.20])
        plan_upgrade_6months = np.random.choice([0, 1], p=[0.85, 0.15])
        
        # Contract renewal
        if contract_type in ['1 Year', '2 Year']:
            contract_duration = 365 if contract_type == '1 Year' else 730
            days_to_contract_end = contract_duration - (tenure_months * 30) % contract_duration
        else:
            days_to_contract_end = 0
        
        # ===== CHURN PROBABILITY CALCULATION =====
        churn_prob = 0.12  # Base churn rate for Thai telecom (slightly higher than global)
        
        # Risk factors that INCREASE churn
        if contract_type == 'Prepaid':
            churn_prob += 0.18  # High prepaid churn in Thailand
        elif contract_type == 'Postpaid Monthly':
            churn_prob += 0.12
        
        if tenure_months < 3:
            churn_prob += 0.25  # New customer risk
        elif tenure_months < 6:
            churn_prob += 0.15
        
        if price_increase_6months > 100:
            churn_prob += 0.20  # Thai customers very price sensitive
        elif price_increase_6months > 0:
            churn_prob += 0.10
        
        if num_support_tickets_6months > 4:
            churn_prob += 0.15
        
        if customer_satisfaction < 2.5:
            churn_prob += 0.25
        elif customer_satisfaction < 3.5:
            churn_prob += 0.12
        
        if monthly_charges > 1500:
            churn_prob += 0.10  # High price sensitivity
        
        if payment_delay_days > 15:
            churn_prob += 0.15
        
        if num_payment_failures > 2:
            churn_prob += 0.12
        
        if online_security == 0 and has_internet:
            churn_prob += 0.05
        
        if last_login_days_ago > 60:
            churn_prob += 0.10  # Low engagement
        
        if province not in ['Bangkok', 'Nonthaburi', 'Pathum Thani', 'Chiang Mai']:
            churn_prob += 0.05  # Less competitive markets
        
        # Protective factors that DECREASE churn
        if tenure_months > 24:
            churn_prob -= 0.20
        elif tenure_months > 12:
            churn_prob -= 0.10
        
        if contract_type == '2 Year':
            churn_prob -= 0.25
        elif contract_type == '1 Year':
            churn_prob -= 0.15
        
        if tech_support == 1:
            churn_prob -= 0.08
        
        if customer_satisfaction > 4.0:
            churn_prob -= 0.15
        elif customer_satisfaction > 3.5:
            churn_prob -= 0.08
        
        if discount_amount > 0:
            churn_prob -= 0.10  # Promotions reduce churn
        
        if multiple_lines >= 2:
            churn_prob -= 0.12  # Family plans sticky
        
        if has_netflix_thai or has_line_tv or has_true_visions:
            churn_prob -= 0.08  # Content bundle reduces churn
        
        if payment_method == 'PromptPay':
            churn_prob -= 0.05  # Easy payment = lower churn
        
        # Cap probability
        churn_prob = max(0.01, min(0.85, churn_prob))
        
        # Determine if churned
        churned = 1 if np.random.random() < churn_prob else 0
        
        # Churn details
        if churned:
            churn_date = account_created + timedelta(days=tenure_months*30)
            # Thai-specific churn reasons
            churn_reason_weights = np.array([0.30, 0.20, 0.18, 0.08, 0.10, 0.08, 0.04, 0.02])
            churn_reason_weights = churn_reason_weights / churn_reason_weights.sum()
            churn_reason = np.random.choice(
                ['Price Too High', 'Poor Signal Quality', 
                 'Competitor Offer', 'Relocated',
                 'Poor Service', 'Better Promotion',
                 'No Longer Need', 'Other'],
                p=churn_reason_weights
            )
            if 'Competitor' in churn_reason or 'Promotion' in churn_reason:
                competitor_switched_to = random.choice([t for t in thai_telcos])
            else:
                competitor_switched_to = None
        else:
            churn_date = None
            churn_reason = None
            competitor_switched_to = None
        
        # ===== CREATE RECORD =====
        record = {
            # IDs & Personal Info
            'customer_id': f'TH{i:07d}',
            'full_name_thai': full_name,
            'gender': gender,
            'age': age,
            'senior_citizen': senior_citizen,
            'partner': partner,
            'dependents': dependents,
            
            # Location (Thailand)
            'province': province,
            'postcode': postcode,
            'region': 'Central' if province in ['Bangkok', 'Nonthaburi', 'Pathum Thani', 'Samut Prakan', 'Nakhon Pathom'] 
                      else 'North' if province in ['Chiang Mai', 'Chiang Rai', 'Lampang', 'Phitsanulok']
                      else 'Northeast' if province in ['Khon Kaen', 'Udon Thani', 'Ubon Ratchathani', 'Nakhon Ratchasima']
                      else 'South',
            
            # Account
            'account_created_date': account_created.strftime('%Y-%m-%d'),
            'tenure_months': tenure_months,
            'contract_type': contract_type,
            'payment_method': payment_method,
            'paperless_billing': paperless_billing,
            
            # Services
            'internet_service': internet_service,
            'online_security': 'Yes' if online_security else 'No',
            'online_backup': 'Yes' if online_backup else 'No',
            'device_protection': 'Yes' if device_protection else 'No',
            'tech_support': 'Yes' if tech_support else 'No',
            'has_netflix_thai': 'Yes' if has_netflix_thai else 'No',
            'has_line_tv': 'Yes' if has_line_tv else 'No',
            'has_true_visions': 'Yes' if has_true_visions else 'No',
            'multiple_lines': multiple_lines,
            
            # Billing (THB)
            'monthly_charges_thb': round(monthly_charges, 2),
            'total_charges_thb': round(total_charges, 2),
            'price_increase_6months_thb': round(price_increase_6months, 2),
            'discount_amount_thb': round(discount_amount, 2),
            'payment_delay_days': payment_delay_days,
            'num_payment_failures': num_payment_failures,
            
            # Usage
            'data_usage_gb': round(data_usage_gb, 2),
            'call_minutes': call_minutes,
            'sms_count': sms_count,
            'social_media_hours_monthly': round(social_media_hours, 2),
            'app_usage_hours_monthly': round(app_usage_hours, 2),
            'website_visits_monthly': website_visits,
            'last_login_days_ago': last_login_days_ago,
            'international_roaming': 'Yes' if international_roaming else 'No',
            'roaming_charges_thb': round(roaming_charges, 2) if international_roaming else 0,
            
            # Customer Service
            'num_support_tickets': num_support_tickets,
            'num_support_tickets_6months': num_support_tickets_6months,
            'avg_ticket_resolution_hours': round(avg_ticket_resolution_hours, 2),
            'num_complaints': num_complaints,
            'customer_satisfaction_score': round(customer_satisfaction, 2),
            'nps_score': nps_score,
            'support_channel_preference': support_channel,
            
            # Product Changes
            'num_plan_changes': num_plan_changes,
            'plan_downgrade_6months': 'Yes' if plan_downgrade_6months else 'No',
            'plan_upgrade_6months': 'Yes' if plan_upgrade_6months else 'No',
            'days_to_contract_end': days_to_contract_end,
            
            # Target Variable
            'churned': churned,
            'churn_date': churn_date.strftime('%Y-%m-%d') if churn_date else None,
            'churn_reason': churn_reason,
            'competitor_switched_to': competitor_switched_to,
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    print(f"\n✓ Generated {len(df):,} Thai customer records")
    print(f"✓ Churn rate: {df['churned'].mean():.2%}")
    print(f"✓ Date range: {df['account_created_date'].min()} to {df['account_created_date'].max()}")
    print(f"✓ Provinces covered: {df['province'].nunique()}")
    print(f"✓ Average monthly charges: ฿{df['monthly_charges_thb'].mean():.2f}")
    
    return df


# ===== GENERATE DATASET =====
print("=" * 60)
print("🇹🇭 THAI CUSTOMER CHURN DATASET GENERATOR")
print("=" * 60)
print()

# Generate 100,000 records (adjust as needed)
df = generate_thai_churn_dataset(n_customers=100000)

# Save to CSV
output_file = 'thai_customer_churn_dataset.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')  # utf-8-sig for Thai characters
print(f"\n✓ Saved to {output_file}")

# ===== DISPLAY SUMMARY =====
print("\n" + "=" * 60)
print("📊 DATASET SUMMARY")
print("=" * 60)

print(f"\n1. SIZE:")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")

print(f"\n2. CHURN STATISTICS:")
print(f"   Total Churned: {df['churned'].sum():,}")
print(f"   Total Active: {(df['churned']==0).sum():,}")
print(f"   Churn Rate: {df['churned'].mean():.2%}")

print(f"\n3. GEOGRAPHIC DISTRIBUTION:")
print(df['province'].value_counts().head(10))

print(f"\n4. CONTRACT TYPES:")
print(df['contract_type'].value_counts())

print(f"\n5. CHURN REASONS (for churned customers):")
churned_df = df[df['churned']==1]
if len(churned_df) > 0:
    print(churned_df['churn_reason'].value_counts())

print(f"\n6. BILLING STATISTICS (THB):")
print(f"   Average Monthly Charges: ฿{df['monthly_charges_thb'].mean():.2f}")
print(f"   Median Monthly Charges: ฿{df['monthly_charges_thb'].median():.2f}")
print(f"   Min: ฿{df['monthly_charges_thb'].min():.2f}")
print(f"   Max: ฿{df['monthly_charges_thb'].max():.2f}")

print(f"\n7. PAYMENT METHODS:")
print(df['payment_method'].value_counts())

print(f"\n8. INTERNET SERVICE:")
print(df['internet_service'].value_counts())

print(f"\n9. SUPPORT CHANNELS:")
print(df['support_channel_preference'].value_counts())

print(f"\n10. MISSING VALUES:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("   Churn-related nulls are intentional (for non-churned customers)")

print("\n" + "=" * 60)
print("✓ SAMPLE RECORDS (First 5)")
print("=" * 60)
print(df.head().to_string())

print("\n" + "=" * 60)
print("✅ DATASET GENERATION COMPLETE!")
print("=" * 60)