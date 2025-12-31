
# 🔓 AWS Security Group Configuration

## Required Inbound Rules for Your EC2 Instance

Add these rules to your EC2 instance's Security Group:

| Type       | Protocol | Port | Source      | Description                    |
|------------|----------|------|-------------|--------------------------------|
| HTTP       | TCP      | 80   | 0.0.0.0/0   | Web interface access          |
| Custom TCP | TCP      | 5000 | 0.0.0.0/0   | Direct API access (optional)  |
| SSH        | TCP      | 22   | Your IP     | SSH access (already configured)|

## How to Configure:

1. **AWS Console** → **EC2** → **Instances**
2. Click your instance → scroll to **Security groups**
3. Click the security group name
4. Go to **Inbound rules** → **Edit inbound rules**
5. **Add rule** for each entry above
6. **Save rules**

## Testing Access:

After deployment, test these endpoints:

```bash
# Replace <EC2_PUBLIC_IP> with your actual EC2 public IP
curl http://<EC2_PUBLIC_IP>/api/health
curl http://<EC2_PUBLIC_IP>
```

