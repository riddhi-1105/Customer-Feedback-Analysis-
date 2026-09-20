"""
Script to generate 300+ realistic sample feedback records.
Run this once to regenerate sample_feedback.csv
"""
import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

POSITIVE_TEMPLATES = [
    "The product quality is excellent. Very happy with my purchase.",
    "Amazing customer service! They resolved my issue within minutes.",
    "Delivery was super fast, got my package in just 2 days.",
    "Great value for money. Highly recommend to everyone.",
    "The product exceeded my expectations. Will definitely buy again.",
    "Very satisfied with my order. Everything was perfect.",
    "Outstanding quality and great packaging. 5 stars!",
    "The app is very smooth and easy to use. Love it.",
    "Excellent product, works exactly as described.",
    "Fast shipping and perfect packaging. Really impressed.",
    "Customer support was extremely helpful and responsive.",
    "The product looks and feels premium. Worth every penny.",
    "Best purchase I've made this year. Absolutely love it.",
    "Very good experience overall. Will order again soon.",
    "The item was delivered on time and in perfect condition.",
    "Happy with the quality. The product is exactly what I needed.",
    "Fantastic experience from ordering to delivery. Thank you!",
    "The website is user-friendly and checkout was smooth.",
    "Got exactly what I ordered. Very pleased with the experience.",
    "Great product at an affordable price. Highly recommended.",
    "The customer service team was very polite and helpful.",
    "My order arrived earlier than expected. Great service!",
    "Product works perfectly. Very happy with the purchase.",
    "Excellent craftsmanship and great customer support.",
    "Quick delivery and product as described. Satisfied!",
    "Loved the packaging and product quality. Will buy again.",
    "The app worked flawlessly. Easy to browse and order.",
    "Really good product. No complaints at all.",
    "Five stars for product quality and delivery speed.",
    "Smooth transaction and quick delivery. Happy customer!",
]

NEGATIVE_DELIVERY = [
    "My order arrived 6 days late. The promised delivery was 2 days.",
    "Package still hasn't arrived after 10 days. Very disappointed.",
    "Delivery was extremely delayed. No tracking updates provided.",
    "My parcel arrived after a week. Unacceptable for paid shipping.",
    "Order was delayed by several days without any notification.",
    "Still waiting for my package after 8 days. Very frustrating.",
    "The delivery timeline was completely wrong. Took 12 days.",
    "My shipment got stuck in transit for 5 days. No support provided.",
    "Very slow delivery. Expected in 3 days but received in 9 days.",
    "Package arrived damaged due to poor shipping handling.",
    "Received a wrong item in my order. Very inconvenient.",
    "The courier did not attempt delivery and left a card.",
    "My order was marked as delivered but I never received it.",
    "Parcel was left unattended outside. Anyone could have taken it.",
    "Delivery was to wrong address. Had to call multiple times.",
    "The package was clearly mishandled. Item was broken inside.",
    "Shipment tracking showed no update for 4 days.",
    "Order was shipped late and the delivery took 2 weeks.",
    "I paid for express delivery but it arrived on standard time.",
    "My package was delivered to a neighbor without my consent.",
]

NEGATIVE_SUPPORT = [
    "Customer service did not respond to my complaint for 3 days.",
    "The support agent was rude and unhelpful.",
    "Waited on hold for 2 hours and still no resolution.",
    "No response to my email even after 5 days.",
    "The chat support kept disconnecting without resolving my issue.",
    "Support promised a callback but never called back.",
    "The customer service team gave wrong information.",
    "I had to contact support 5 times for the same issue.",
    "The helpline number is always busy. Very poor service.",
    "Support agent was dismissive and did not take my complaint seriously.",
    "My complaint was closed without resolution.",
    "The support team took 7 days to respond to my simple question.",
    "Customer service is the worst I have ever experienced.",
    "No one from support is following up on my case.",
    "The agent couldn't resolve my issue and transferred me 3 times.",
    "Support chat is slow and agents don't seem knowledgeable.",
    "I've been waiting 4 days for a response to my support ticket.",
    "The phone support team put me on hold for 90 minutes.",
    "Support response was generic and didn't address my problem.",
    "Very poor customer service. Will not recommend this company.",
]

NEGATIVE_PAYMENT = [
    "Money was deducted from my account but order was cancelled.",
    "I was charged twice for the same order.",
    "Payment failed but money was still deducted.",
    "Extra charges appeared on my bill that were not mentioned.",
    "Refund has not been processed even after 15 days.",
    "The payment gateway keeps showing errors.",
    "My card was charged for a product I never ordered.",
    "Billing amount was higher than the price shown at checkout.",
    "Money deducted but I received no order confirmation.",
    "Still waiting for my refund after filing the request 3 weeks ago.",
    "The EMI charges were wrong and I was overcharged.",
    "Payment was processed twice and I need a refund.",
    "Hidden charges added at checkout without prior notice.",
    "Refund was promised within 5 days but it's been 2 weeks.",
    "The invoice does not match the actual amount charged.",
]

NEGATIVE_QUALITY = [
    "The product stopped working after just 2 days of use.",
    "Very poor build quality. The item broke within a week.",
    "This product is completely defective. Does not work at all.",
    "The quality is terrible. Not worth the money.",
    "Product was damaged when I received it. Clearly poor packaging.",
    "The material feels cheap and is already peeling.",
    "This item is nothing like described. Very disappointed.",
    "Product quality is far below what was advertised.",
    "The color is completely different from what was shown online.",
    "Battery life is terrible. Lasts only 2 hours.",
    "The product developed a fault within the first week.",
    "Very flimsy construction. Would not recommend.",
    "Size is wrong. The product is much smaller than advertised.",
    "The finish is poor and shows scratches easily.",
    "Product smell is very strong and unpleasant.",
]

NEGATIVE_REFUND = [
    "Requested a return 10 days ago and still waiting for pickup.",
    "The return process is extremely complicated.",
    "My refund was rejected without any valid reason.",
    "It's been 3 weeks and my refund still hasn't arrived.",
    "The return label they sent was invalid.",
    "Unable to initiate a return through the website. It keeps erroring.",
    "Refund amount received was less than what I paid.",
    "My return request was ignored for 2 weeks.",
    "The customer service denied my refund claim unfairly.",
    "Returning a product is very difficult with this company.",
]

NEGATIVE_APP = [
    "The app crashes every time I try to place an order.",
    "The website is very slow and keeps timing out.",
    "Cannot complete checkout on the app. It shows an error.",
    "The mobile app is full of bugs. Very frustrating to use.",
    "Search functionality on the website doesn't work properly.",
    "The app logged me out and deleted my cart.",
    "Website interface is confusing and hard to navigate.",
    "Cannot access my order history on the app.",
    "The app keeps showing loading but never loads the page.",
    "Payment page on the app freezes every time.",
]

NEGATIVE_ACCOUNT = [
    "Cannot login to my account even with correct password.",
    "My account was locked without any explanation.",
    "OTP is not being received on my registered number.",
    "Password reset link is not working.",
    "My order history has disappeared from my account.",
    "Cannot update my delivery address on the account page.",
    "My account was hacked and I need urgent help.",
    "The verification process is too complicated.",
]

NEUTRAL_TEMPLATES = [
    "Delivery was on time but the packaging could be better.",
    "Product is okay. Nothing special but does the job.",
    "The customer service was average. Could be more helpful.",
    "Item received as described. No major complaints.",
    "Product is decent for the price.",
    "Shipping took longer than expected but product is fine.",
    "The app is functional but needs some improvements.",
    "Product quality is acceptable. Not great not bad.",
    "Order arrived correctly but the delivery experience was average.",
    "Overall experience was satisfactory. Could be improved.",
    "The product meets basic requirements.",
    "Received my order. It is as described.",
    "Customer support responded eventually but took time.",
    "Product works but battery life could be better.",
    "Packaging was simple but item was protected.",
    "Average experience. Would try again to see if it improves.",
    "The website is functional but a bit outdated.",
    "Order delivered correctly. No major issues to report.",
    "Product is acceptable for the price point.",
    "Service was standard. Nothing exceptional.",
]

PRODUCTS = ["Bluetooth Headphones", "Wireless Mouse", "Laptop Stand", "USB Hub",
            "Phone Case", "Power Bank", "Smart Watch", "Keyboard", "Webcam",
            "HDMI Cable", "Desk Lamp", "Phone Charger", "Screen Protector",
            "Laptop Bag", "Mechanical Keyboard", "Monitor", "Earbuds", "Router"]

CHANNELS = ["Website", "Mobile App", "WhatsApp", "Phone", "Email", "In-Store"]

start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

rows = []
feedback_id = 1

def random_date():
    delta = end_date - start_date
    return start_date + timedelta(days=random.randint(0, delta.days))

def add_rows(templates, n, rating_range):
    global feedback_id
    for i in range(n):
        tmpl = random.choice(templates)
        # Add slight variation
        variation = random.choice([
            "", " Would {}.".format(random.choice(["not buy again", "recommend", "try again"])),
            " Overall {}% satisfied.".format(random.randint(20, 100)),
            ""
        ])
        rows.append({
            "feedback_id": feedback_id,
            "date": random_date().strftime("%Y-%m-%d"),
            "customer_id": f"CUST{random.randint(1000, 9999)}",
            "feedback": tmpl + variation,
            "rating": random.randint(*rating_range),
            "product": random.choice(PRODUCTS),
            "channel": random.choice(CHANNELS),
        })
        feedback_id += 1

# Distribution: ~30% positive, ~15% neutral, ~55% negative (various types)
add_rows(POSITIVE_TEMPLATES, 90, (4, 5))
add_rows(NEUTRAL_TEMPLATES, 45, (3, 4))
add_rows(NEGATIVE_DELIVERY, 50, (1, 2))
add_rows(NEGATIVE_SUPPORT, 40, (1, 2))
add_rows(NEGATIVE_PAYMENT, 35, (1, 2))
add_rows(NEGATIVE_QUALITY, 35, (1, 2))
add_rows(NEGATIVE_REFUND, 20, (1, 2))
add_rows(NEGATIVE_APP, 20, (1, 2))
add_rows(NEGATIVE_ACCOUNT, 15, (1, 2))

# Shuffle
random.shuffle(rows)
for i, row in enumerate(rows, 1):
    row["feedback_id"] = i

df = pd.DataFrame(rows)
print(f"Generated {len(df)} records")
print(df.head())
df.to_csv("data/sample_feedback.csv", index=False)
print("Saved to data/sample_feedback.csv")
