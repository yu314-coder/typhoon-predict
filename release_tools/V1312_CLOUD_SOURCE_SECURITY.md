# 1.3.12 native-source transport, not a training bundle

This manual-only CPU job retrieves all 81,678 exact native CMORPH source hours
from the frozen 13,613-target plan. Packing changes neither cases, source
resolution, native bytes, genuine missing cells nor true dry zeros. It does
not decode scientific labels, train, publish a website or change live workflows.

The authorized parent is the existing typhoon_predict Drive project. All new
uploads are under training_bundles/1.3.12/source-staging/cloud-cmorph-native-20261009.
An authenticated request on the Mac reserves each NEW file's name and parent.
The cloud job receives only that file's resumable-upload session, NOT a Google
access/refresh token, rclone configuration, account credential or personal
GitHub token. These capabilities cannot enumerate, read, move or delete other
Drive files, and cannot change the parent fixed by the reservation. Completed
upload responses must match the expected name, parent, size, MD5 and available
server SHA256. Native-source SHA256 receipts remain inside each archive.

Setup uses the existing local GitHub login. No new personal GitHub token is
required. The built-in job token has contents-read permission only; checkout
is pinned and does not persist credentials. The source code, tests and public
transport plan are hash-checked BEFORE exposing capability secrets to the
download step. Only the repository owner's manual dispatch on main is allowed.
The separate environment permits only main, and one concurrency group prevents
duplicate writers. No third-party data decoder or package installation runs.

After the cloud process confirms it loaded every capability, the setup operator
must DELETE all eight environment secrets and independently verify the empty
secret list before announcing that the Mac can shut down. The running process
already holds its capabilities in memory and can continue without the Mac.
At process/VM termination that memory disappears; Drive upload sessions also
have a finite provider lifetime. Capabilities are never written to logs, public
Git, artifacts or runner files. Completed archives remain in Drive.

This is one bounded job (350-minute timeout, source retries and scratch limits),
not indefinite automatic recovery. An interrupted or failed job preserves
completed Drive archives but requires reviewed continuation and new file
reservations. Removing stored secrets deliberately prevents unattended reruns.
The final receipt is written ONLY after all 81,678 native hours and archive
checksums are committed. Full native decode, twelve-slot interval/mask/area
audits, final corpus reconciliation, CUDA trainer and notebook readiness remain
separate unfinished gates. No source-download receipt means training-ready.
