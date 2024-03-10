import paramiko
from scp import SCPClient

def create_ssh_client(server_address, port, username, password):
    """Create SSH client session to remote server"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server_address, port, username, password)
        return ssh
    except Exception as e:
        print(f"Connection Failed: {e}")
        return None

def send_file(ssh, local_path, remote_path):
    """Send file to remote server"""
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(local_path, remote_path)

def receive_file(ssh, remote_path, local_path):
    """Receive file from remote server"""
    with SCPClient(ssh.get_transport()) as scp:
        scp.get(remote_path, local_path)

# Example usage
if __name__ == "__main__":
    server_num = '02'
    server = 'sp24-cs525-05' + server_num + '.cs.illinois.edu'
    port = 22  # Default SSH port
    username = 'your_username'
    password = 'your_password'
    
    ssh = create_ssh_client(server, port, username, password)
    
    if ssh:
        src_img_path = '.../ILSVRC2012_val_00000001.JPEG' # insert your path to image here
        dst_img_path = '' # insert your destination path to image here
        send_file(ssh, src_img_path, dst_img_path)
        print("File sent successfully.")
        
        # receive_file(ssh, 'path/on/remote/image.jpg', 'path/to/local/received_image.jpg')
        # print("File received successfully.")
        
        ssh.close()
