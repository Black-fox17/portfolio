import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import { blogPosts } from '../constants/blogData';
import Navbar from './Navbar';
import './BlogStyles.css';

const BlogPost = () => {
    const { id } = useParams<{ id: string }>();
    const [content, setContent] = useState('');
    const navigate = useNavigate();

    const post = blogPosts.find((p) => p.id === id);

    useEffect(() => {
        post?.file().then((res) => setContent(res.default));
    }, [post]);


    if (!post) {
        return (
            <div className="blog-container">
                <Navbar />
                <div className="blog-content">
                    <div className="blog-not-found">
                        <h1>Blog post not found</h1>
                        <button onClick={() => navigate('/blog')} className="back-button">
                            ← Back to Blog
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="blog-container">
            <Navbar />

            <div className="blog-content">
                <article className="blog-post">
                    <button onClick={() => navigate('/blog')} className="back-button">
                        ← Back to Blog
                    </button>

                    <header className="blog-post-header">
                        <span className="blog-category">{post.category}</span>
                        <h1 className="blog-post-title">{post.title}</h1>

                        <div className="blog-post-meta">
                            <span className="blog-post-author">Ayeleru Abdulsalam Oluwaseun</span>
                            <span className="blog-meta-separator">•</span>
                            <time className="blog-post-date">
                                {new Date(post.date).toLocaleDateString('en-US', {
                                    year: 'numeric',
                                    month: 'long',
                                    day: 'numeric'
                                })}
                            </time>
                            <span className="blog-meta-separator">•</span>
                            <span className="blog-post-read-time">{post.readTime}</span>
                        </div>
                    </header>

                    <div className="blog-post-content">
                        <ReactMarkdown
                        
                            remarkPlugins={[remarkMath, remarkGfm]}
                            rehypePlugins={[rehypeKatex, rehypeRaw]}
                            components={{
                                code({ node, inline, className, children, ...props }) {
                                    return inline ? (
                                        <code className={className} {...props}>
                                            {children}
                                        </code>
                                    ) : (
                                        <code className={className} {...props}>
                                            {children}
                                        </code>
                                    );
                                },
                            }}
                        >
                            {content}
                        </ReactMarkdown>
                    </div>

                    <footer className="blog-post-footer">
                        <button onClick={() => navigate('/blog')} className="back-button-footer">
                            ← Back to all posts
                        </button>
                    </footer>
                </article>
            </div>
        </div>
    );
};

export default BlogPost;
