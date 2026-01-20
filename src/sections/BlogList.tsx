import React from 'react';
import { useNavigate } from 'react-router-dom';
import { blogPosts } from '../constants/blogData';
import Navbar from './Navbar';
import { Helmet } from 'react-helmet-async';
import './BlogStyles.css';

const BlogList = () => {
    const navigate = useNavigate();

    const handleBlogClick = (id: string) => {
        navigate(`/blog/${id}`);
    };

    return (
        <div className="blog-container">
            <Navbar />

            <div className="blog-content">
                <Helmet>
                    <title>The Deep End - Blog | Ayeleru Abdulsalam</title>
                    <meta name="description" content="Thoughts on mathematics, deep learning, computer vision, large language models, and backend architecture by Ayeleru Abdulsalam." />
                    <link rel="canonical" href="https://salam-portfolio-three.vercel.app/blog" />

                    <meta property="og:type" content="website" />
                    <meta property="og:url" content="https://salam-portfolio-three.vercel.app/blog" />
                    <meta property="og:title" content="The Deep End - Blog | Ayeleru Abdulsalam" />
                    <meta property="og:description" content="Thoughts on mathematics, deep learning, computer vision, large language models, and backend architecture." />
                </Helmet>
                <header className="blog-header">
                    <h1 className="blog-title">The Deep End</h1>
                    <p className="blog-subtitle">
                        Thoughts on mathematics, deep learning, computer vision, large language models, and backend architecture
                    </p>
                    <div className="blog-author-info">
                        <p className="author-name">by <a href="mailto:ayeleru1234@gmail.com">Ayeleru Abdulsalam Oluwaseun</a></p>
                    </div>
                </header>

                <div className="blog-grid">
                    {blogPosts.map((post) => (
                        <article
                            key={post.id}
                            className="blog-card"
                            onClick={() => handleBlogClick(post.id)}
                        >
                            <div className="blog-card-header">
                                <span className="blog-category">{post.category}</span>
                                <span className="blog-read-time">{post.readTime}</span>
                            </div>

                            <h2 className="blog-card-title">{post.title}</h2>

                            <p className="blog-card-excerpt">{post.excerpt}</p>

                            <div className="blog-card-footer">
                                <time className="blog-date">{new Date(post.date).toLocaleDateString('en-US', {
                                    year: 'numeric',
                                    month: 'long',
                                    day: 'numeric'
                                })}</time>
                                <span className="blog-read-more">Read more →</span>
                            </div>
                        </article>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default BlogList;
