
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { getDatabase } = require('./src/_data/mongodb');
const fs = require('fs').promises;
const path = require('path');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(bodyParser.json());
app.use(express.static('admin'));

// Authentication middleware
const authenticate = (req, res, next) => {
  const { password } = req.headers;
  if (password !== 'towardsai2025') {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
};

// Get all posts
app.get('/api/posts', authenticate, async (req, res) => {
  try {
    const db = await getDatabase();
    const posts = await db.collection('posts').find({}).toArray();
    res.json(posts);
  } catch (error) {
    console.error('Error fetching posts:', error);
    res.status(500).json({ error: 'Failed to fetch posts' });
  }
});

// Get single post
app.get('/api/posts/:id', authenticate, async (req, res) => {
  try {
    const db = await getDatabase();
    const { ObjectId } = require('mongodb');
    const post = await db.collection('posts').findOne({ _id: new ObjectId(req.params.id) });
    if (!post) {
      return res.status(404).json({ error: 'Post not found' });
    }
    res.json(post);
  } catch (error) {
    console.error('Error fetching post:', error);
    res.status(500).json({ error: 'Failed to fetch post' });
  }
});

// Create new post
app.post('/api/posts', authenticate, async (req, res) => {
  try {
    const db = await getDatabase();
    const postData = {
      ...req.body,
      date: new Date(req.body.date || Date.now()),
      createdAt: new Date(),
      updatedAt: new Date()
    };

    const result = await db.collection('posts').insertOne(postData);
    
    // Create markdown file
    await createMarkdownFile(postData);
    
    res.json({ id: result.insertedId, ...postData });
  } catch (error) {
    console.error('Error creating post:', error);
    res.status(500).json({ error: 'Failed to create post' });
  }
});

// Update post
app.put('/api/posts/:id', authenticate, async (req, res) => {
  try {
    const db = await getDatabase();
    const { ObjectId } = require('mongodb');
    const postData = {
      ...req.body,
      updatedAt: new Date()
    };

    const result = await db.collection('posts').updateOne(
      { _id: new ObjectId(req.params.id) },
      { $set: postData }
    );

    if (result.matchedCount === 0) {
      return res.status(404).json({ error: 'Post not found' });
    }

    // Update markdown file
    await createMarkdownFile({ ...postData, _id: req.params.id });
    
    res.json({ id: req.params.id, ...postData });
  } catch (error) {
    console.error('Error updating post:', error);
    res.status(500).json({ error: 'Failed to update post' });
  }
});

// Delete post
app.delete('/api/posts/:id', authenticate, async (req, res) => {
  try {
    const db = await getDatabase();
    const { ObjectId } = require('mongodb');
    
    // Get post data first
    const post = await db.collection('posts').findOne({ _id: new ObjectId(req.params.id) });
    if (!post) {
      return res.status(404).json({ error: 'Post not found' });
    }

    // Delete from database
    await db.collection('posts').deleteOne({ _id: new ObjectId(req.params.id) });
    
    // Delete markdown file
    await deleteMarkdownFile(post.slug);
    
    res.json({ message: 'Post deleted successfully' });
  } catch (error) {
    console.error('Error deleting post:', error);
    res.status(500).json({ error: 'Failed to delete post' });
  }
});

// Helper function to create markdown files
async function createMarkdownFile(postData) {
  const slug = postData.slug || postData.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
  const filename = `${slug}.md`;
  const filePath = path.join(__dirname, 'src', 'posts', filename);

  const frontMatter = `---
title: "${postData.title}"
date: ${postData.date.toISOString()}
author: "${postData.author}"
excerpt: "${postData.excerpt}"
tags: [${postData.tags ? postData.tags.map(tag => `"${tag}"`).join(', ') : '"AI"'}]
featured_image: "${postData.featured_image || ''}"
seo_title: "${postData.seo_title || postData.title}"
seo_description: "${postData.seo_description || postData.excerpt}"
---

${postData.body || postData.content || ''}
`;

  await fs.writeFile(filePath, frontMatter);
  console.log(`Created/updated markdown file: ${filename}`);
}

// Helper function to delete markdown files
async function deleteMarkdownFile(slug) {
  const filename = `${slug}.md`;
  const filePath = path.join(__dirname, 'src', 'posts', filename);
  
  try {
    await fs.unlink(filePath);
    console.log(`Deleted markdown file: ${filename}`);
  } catch (error) {
    console.error(`Error deleting markdown file: ${filename}`, error);
  }
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Admin API server running on http://0.0.0.0:${PORT}`);
  console.log(`🚀 Admin panel available at: /admin/dynamic-admin.html`);
  console.log(`🔑 Password: towardsai2025`);
});
