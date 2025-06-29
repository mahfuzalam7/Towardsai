# Towards AI Blog - Replit Configuration

## Overview

Towards AI is a production-ready, fully responsive blog website built with Eleventy (11ty). It features a sleek, futuristic dark mode design with glassmorphic UI elements, designed for AI content monetization. The site serves as a platform for publishing cutting-edge insights into artificial intelligence, machine learning, and future technology trends.

## System Architecture

### Frontend Architecture
- **Static Site Generator**: Eleventy (11ty) v3.1.2 for lightning-fast performance
- **Templating Engine**: Nunjucks for dynamic content rendering
- **Styling**: Modern CSS with CSS Grid/Flexbox layouts, custom properties, and glassmorphic design
- **Typography**: Custom font stack using Inter, IBM Plex Sans, and Space Grotesk
- **Responsive Design**: Mobile-first approach with 1 column on phones, 2 on tablets, 3+ on desktop

### Backend Architecture
- **Build System**: Node.js 20 with npm package management
- **Content Management**: File-based content with Markdown and front matter
- **Static Generation**: Pre-built HTML/CSS/JS served statically
- **Asset Processing**: Pass-through copying for images and static assets

### Content Management System
- **Netlify CMS**: Web-based admin interface at `/admin/`
- **Authentication**: Git-gateway with GitHub integration
- **Content Types**: Blog posts, author profiles, and static pages
- **Media Management**: Image uploads to `src/images/uploads/`

## Key Components

### Core Pages
- **Homepage** (`src/index.njk`): Featured posts grid with hero section
- **Blog Posts** (`src/posts/*.md`): Individual articles with rich metadata
- **Author Pages** (`src/authors/*.md`): Author profiles with social links and post listings
- **Static Pages**: About (`src/about.njk`) and Contact (`src/contact.njk`) pages

### Layout System
- **Base Layout** (`src/_includes/layouts/base.njk`): Core HTML structure with SEO meta tags
- **Post Layout** (`src/_includes/layouts/post.njk`): Article layout with author info and affiliate links
- **Page Layout** (`src/_includes/layouts/page.njk`): Static page template
- **Author Layout** (`src/_includes/layouts/author.njk`): Author profile template

### Reusable Components
- **Header** (`src/_includes/components/header.njk`): Navigation with mobile menu
- **Footer** (`src/_includes/components/footer.njk`): Site links and social media
- **Post Card** (`src/_includes/components/post-card.njk`): Blog post preview component
- **Newsletter** (`src/_includes/components/newsletter.njk`): Email subscription form
- **Ad Zone** (`src/_includes/components/ad-zone.njk`): Advertisement placement system

### Monetization Features
- **Affiliate Links**: Structured affiliate link sections in posts
- **Ad Placement**: Multiple ad zones (header, sidebar, in-content, footer)
- **Newsletter Integration**: Built-in subscription forms
- **Social Sharing**: Twitter and LinkedIn share buttons

## Data Flow

### Content Creation Flow
1. Authors create content via Netlify CMS admin panel or direct markdown files
2. Eleventy processes markdown files during build process
3. Content is enriched with metadata (reading time, excerpts, date formatting)
4. Static HTML pages are generated with optimized performance

### Build Process
1. Eleventy reads configuration from `.eleventy.js`
2. Processes content files from `src/` directory
3. Applies layouts and includes from `src/_includes/`
4. Copies static assets from `assets/` and `src/images/`
5. Generates output to `_site/` directory

### Data Sources
- **Site Configuration**: `src/_data/site.json` for global settings
- **Author Data**: `src/_data/authors.json` for author information
- **Posts Collection**: Auto-generated from `src/posts/*.md` files
- **Authors Collection**: Auto-generated from `src/authors/*.md` files

## External Dependencies

### Core Dependencies
- **@11ty/eleventy**: Static site generator and build system
- **luxon**: Date/time formatting and manipulation
- **markdown-it**: Markdown processing with extensions
- **markdown-it-anchor**: Automatic heading anchors
- **markdown-it-prism**: Syntax highlighting for code blocks

### Content Management
- **Netlify CMS**: Web-based content management interface
- **Netlify Identity**: Authentication for admin access
- **Git-gateway**: Content version control integration

### Third-party Integrations
- **Google Fonts**: Typography (Inter, IBM Plex Sans, Space Grotesk)
- **Font Awesome**: Icon system for UI elements
- **Google Analytics**: Traffic analytics (configurable)
- **Ad Networks**: Support for Google AdSense, Carbon Ads, Media.net
- **Email Providers**: Newsletter integration options

## Deployment Strategy

### Netlify Configuration
- **Build Command**: `npx @11ty/eleventy`
- **Publish Directory**: `_site`
- **Node Version**: 18 (specified in netlify.toml)
- **Security Headers**: X-Frame-Options, XSS-Protection, Content-Type-Options
- **Caching**: Long-term caching for static assets

### Performance Optimizations
- **Static Generation**: Pre-built pages for fast loading
- **Asset Optimization**: Efficient font loading and image handling
- **Progressive Enhancement**: Works without JavaScript
- **SEO Features**: Sitemap, RSS feed, Open Graph, Twitter Cards

### Development Workflow
- **Local Development**: `npm run serve` for live reloading
- **Build Process**: `npm run build` for production builds
- **Version Control**: Git-based workflow with Netlify deployment

## Changelog

```
Changelog:
- June 27, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```