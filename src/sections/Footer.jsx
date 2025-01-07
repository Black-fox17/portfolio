const Footer = () => {
    return (
      <footer className="c-space pt-7 pb-3 border-t border-black-300 flex justify-between items-center flex-wrap gap-5">
        <div className="text-white-500 flex gap-2">
          <p>Thanks For checking Me out</p>
          <p>|</p>
          <p>Opened to work</p>
        </div>
  
        <div className="flex gap-3">
          <div className="social-icon">
            <a 
            href="https://github.com/Black-fox17"
            target="_blank">
                <img src="/assets/github.svg" alt="github" className="w-1/2 h-1/2 ml-3" />
            </a>
          </div>
          <div className="social-icon">
            <a
            href="https://x.com/Ayeleru_Salam"
            target="_blank">
                <img src="/assets/twitter.svg" alt="twitter" className="w-1/2 h-1/2 flex ml-3" />
            </a>
          </div>
        </div>
  
        <p className="text-white-500">© 2025 Ayeleru Abdulsalam. All rights reserved.</p>
      </footer>
    );
  };
  
  export default Footer;