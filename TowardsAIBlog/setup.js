
const { connectToDatabase } = require('./src/_data/mongodb');

async function setup() {
  try {
    console.log('Setting up MongoDB connection...');
    await connectToDatabase();
    console.log('✅ MongoDB connected successfully!');
    
    console.log('🚀 Setup complete! You can now:');
    console.log('1. Run "node sync-posts.js" to sync existing posts to MongoDB');
    console.log('2. Start the admin server with "node admin-server.js"');
    console.log('3. Visit http://localhost:3001/dynamic-admin.html for the admin panel');
    console.log('4. Password: towardsai2025');
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Setup failed:', error.message);
    process.exit(1);
  }
}

setup();
