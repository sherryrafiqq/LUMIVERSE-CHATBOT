# 🔧 Supabase Setup Guide for Production

This guide will help you set up Supabase properly for production use with Row Level Security (RLS) enabled.

## 📋 Prerequisites

1. **Supabase Project** - You already have this
2. **Service Role Key** - You need to get this from your Supabase dashboard

## 🔑 Getting Your Service Role Key

### Step 1: Access Supabase Dashboard
1. Go to [supabase.com](https://supabase.com)
2. Sign in to your account
3. Select your project: `qvqbhoptpecvflidiqik`

### Step 2: Get the Service Role Key
1. In your project dashboard, go to **Settings** (gear icon)
2. Click on **API** in the left sidebar
3. Scroll down to **Project API keys**
4. Copy the **service_role** key (NOT the anon key)

### Step 3: Add to Environment Variables
1. Open your `variables.env` file
2. Add your service role key:
   ```
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
   ```

## 🛡️ Row Level Security (RLS) Policies

Your database already has RLS enabled, which is good for security. Here are the policies you need:

### Users Table Policies

```sql
-- Allow users to read their own profile
CREATE POLICY "Users can view own profile" ON users
FOR SELECT USING (auth.uid() = id);

-- Allow service role to insert users (for registration)
CREATE POLICY "Service role can insert users" ON users
FOR INSERT WITH CHECK (auth.role() = 'service_role');

-- Allow service role to update users
CREATE POLICY "Service role can update users" ON users
FOR UPDATE USING (auth.role() = 'service_role');
```

### Emotion Logs Table Policies

```sql
-- Allow users to insert their own emotion logs
CREATE POLICY "Users can insert own emotion logs" ON emotion_logs
FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Allow users to read their own emotion logs
CREATE POLICY "Users can view own emotion logs" ON emotion_logs
FOR SELECT USING (auth.uid() = user_id);

-- Allow service role to insert emotion logs
CREATE POLICY "Service role can insert emotion logs" ON emotion_logs
FOR INSERT WITH CHECK (auth.role() = 'service_role');
```

## 🚀 How the System Works

### User Registration Flow
1. **User sends:** `register:username`
2. **System creates user** through Supabase Auth using service role
3. **User profile inserted** into `users` table
4. **Registration logged** in `emotion_logs` table

### Chat Flow
1. **User sends message**
2. **System looks up user** by username in `users` table
3. **AI processes message** and detects emotion
4. **Response generated** and sent to user
5. **Interaction logged** in `emotion_logs` table

## 🔒 Security Benefits

- **RLS enabled** - Users can only access their own data
- **Service role** - Only your app can create users
- **Auth integration** - Users are properly authenticated
- **Data isolation** - Each user's data is protected

## ⚠️ Important Notes

1. **Keep service role key secret** - Never expose it in client-side code
2. **Use environment variables** - Always use `variables.env` for local development
3. **Railway deployment** - Add `SUPABASE_SERVICE_ROLE_KEY` to Railway environment variables
4. **Test thoroughly** - Verify user creation and logging work correctly

## 🧪 Testing

After setup, test your system:

1. **Register a user:** `register:test_user`
2. **Send a message:** `hey`
3. **Check Supabase dashboard** - Verify data appears in both tables
4. **Check RLS** - Try accessing data from different users

## 🆘 Troubleshooting

### "Service role key not found"
- Make sure you've added `SUPABASE_SERVICE_ROLE_KEY` to your environment variables
- Check that the key is correct (starts with `eyJ...`)

### "RLS policy violation"
- Verify the SQL policies above are applied to your tables
- Check that the service role is being used for admin operations

### "User creation failed"
- Ensure Supabase Auth is enabled in your project
- Check that the service role has proper permissions

## ✅ Success Indicators

- ✅ Users can register with `register:username`
- ✅ Chat messages are logged to `emotion_logs`
- ✅ Each user can only see their own data
- ✅ No RLS policy violations in logs
- ✅ Service role key is properly configured
