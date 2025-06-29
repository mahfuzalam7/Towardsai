
const fs = require('fs').promises;
const path = require('path');
const { getDatabase } = require('./src/_data/mongodb');

async function syncPostsToMongoDB() {
  try {
    console.log('Connecting to MongoDB...');
    const db = await getDatabase();
    
    console.log('Reading existing posts...');
    const postsDir = path.join(__dirname, 'src', 'posts');
    const files = await fs.readdir(postsDir);
    const markdownFiles = files.filter(file => file.endsWith('.md'));
    
    for (const file of markdownFiles) {
      const filePath = path.join(postsDir, file);
      const content = await fs.readFile(filePath, 'utf-8');
      
      // Parse front matter
      const frontMatterMatch = content.match(/^---\n([\s\S]+?)\n---\n([\s\S]*)$/);
      if (!frontMatterMatch) continue;
      
      const frontMatter = frontMatterMatch[1];
      const body = frontMatterMatch[2];
      
      // Parse YAML-like front matter
      const postData = {};
      frontMatter.split('\n').forEach(line => {
        const match = line.match(/^(\w+):\s*(.+)$/);
        if (match) {
          const [, key, value] = match;
          if (key === 'tags') {
            postData[key] = value.replace(/[\[\]"]/g, '').split(', ').filter(tag => tag.trim());
          } else if (key === 'date') {
            postData[key] = new Date(value);
          } else {
            postData[key] = value.replace(/^["']|["']$/g, '');
          }
        }
      });
      
      postData.body = body.trim();
      postData.content = body.trim();
      postData.slug = file.replace('.md', '');
      postData.createdAt = new Date();
      postData.updatedAt = new Date();
      
      // Check if post already exists
      const existingPost = await db.collection('posts').findOne({ slug: postData.slug });
      
      if (existingPost) {
        console.log(`Updating existing post: ${postData.title}`);
        await db.collection('posts').updateOne(
          { slug: postData.slug },
          { $set: postData }
        );
      } else {
        console.log(`Adding new post: ${postData.title}`);
        await db.collection('posts').insertOne(postData);
      }
    }
    
    console.log('Posts synced successfully!');
    process.exit(0);
  } catch (error) {
    console.error('Error syncing posts:', error);
    process.exit(1);
  }
}

syncPostsToMongoDB();
